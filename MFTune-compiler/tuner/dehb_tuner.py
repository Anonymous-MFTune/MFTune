import random
import time
from systems.gcc_compiler import GccCompiler
from systems.clang_compiler import ClangCompiler
from workload import WorkloadController
from tqdm import tqdm
from utils.logger import Logger
from math import log, ceil
import numpy as np


class DEHBTuner:
    def __init__(self, args_compiler, args_workload, args_tune, run):
        super(DEHBTuner, self).__init__()
        self.max_iter = int(args_tune['max_iter'])
        self.total_budget = int(args_tune['total_budget'])
        self.args_compiler = args_compiler
        self.args_workload = args_workload
        self.args_tune = args_tune
        self.workload_bench = args_workload["workload_bench"]
        self.tuning_method = args_tune['tuning_method']
        self.fidelity_type = args_tune['fidelity_type']
        self.fidelity_metric = args_tune['fidelity_metric']
        self.optimize_objective = args_tune['optimize_objective']
        self.sys_name = self.args_compiler['compiler']

        self.log_path = f"experimental_results/{self.sys_name}/{self.workload_bench}/{self.tuning_method}/run_{run}_{self.tuning_method}_{self.fidelity_type}"
        self.log_file = 'DEHBTuner_results.csv'
        self.cyber_twin_path = f"experimental_results/{self.sys_name}/{self.workload_bench}"
        self.cyber_twin_file = 'cyber-twin.csv'

        # Target system
        if self.sys_name == 'gcc':
            self.target_system = GccCompiler(args_compiler)
        elif self.sys_name == 'clang':
            self.target_system = ClangCompiler(args_compiler)

        self.workload_controller = WorkloadController(args_compiler, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)

        self.knobs_info = self.target_system.knobs_info
        self.config_dim = len(self.knobs_info)

        if self.optimize_objective in ['latency', 'run_time']:
            self.is_minimize = True
        elif self.optimize_objective in ['throughput', 'RPS']:
            self.is_minimize = False
        else:
            raise ValueError(f"Unknown optimization objective: {self.optimize_objective}")

        self.f = 0.5   # scaling factor (for DE mutant)
        self.cr = 0.5  # crossover rate (for DE crossover)
        self.eta = 2   # down-sampling rate (for HS)
        self.default_factors = self.workload_controller.get_default_fidelity_factors()  # full-fidelity settings
        self.R_unit = 5  # Define resource unit
        self.hb_factor = 'max-funcs'
        self.R = self.default_factors[self.hb_factor] / self.R_unit  # Maximum resource unit per config (only time factor)
        self.s_max = int(log(self.R) / log(self.eta))   # Maximum stage in HS, as well as maximum bracket
        self.B = (self.s_max + 1) * self.R  # budget per bracket, i.e., budget for each full HS process

        self.subpopulations = {}  # current subpopulation under each budget level
        self.global_pool = {}   # global pop pool, recording all configs under each budget level
        self.history = []
        self.hf_evaluated_configs = []
        self.consumed_cost = 0

        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")
        print()

    def tune_dehb(self):
        start_time = time.time()
        iteration = 1
        while self.consumed_cost < self.total_budget:
            print(f"Executing the {iteration}-th iteration of DEHB ...")
            # Outer loop (Bracket)
            for s in reversed(range(self.s_max + 1)):
                n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
                r = self.R * self.eta ** (-s)
                # Inner loop (Successive Halving)
                for i in range(s + 1):
                    n_i = int(n * self.eta ** (-i))
                    r_i = int(self.R) if i == s else int(r * self.eta ** i)

                    budget = r_i * self.R_unit
                    factors = self.default_factors.copy()
                    factors[self.hb_factor] = budget
                    prev_budget = int(r_i / self.eta) * self.R_unit if i > 0 else None

                    print(f"[Iteration: {iteration}] | [Bracket: {self.s_max - s + 1}] | [SH stage: {i+1}] | [Configs: {n_i}] | [Budget: {budget}s]")

                    # first bracket of first iteration at first stage [iter-1; bracket-1; stage-1]
                    is_first = (s == self.s_max and i == 0 and not self.subpopulations)

                    if is_first:
                        # iter-1; bracket-1; stage-1; --> random sampling...
                        configs = [self.random_sample_one() for _ in range(n_i)]
                        results, _ = self.evaluate_configs(configs, factors)
                        vectors = [self.encode_config(cfg) for cfg, _, _ in results]
                        self.subpopulations[budget] = vectors
                        self.global_pool.setdefault(budget, []).extend(vectors)
                        for (cfg, perf, _), vec in zip(results, vectors):
                            self.history.append((vec, perf, budget))
                        if budget == self.R * self.R_unit:
                            self.hf_evaluated_configs.extend(results)
                        continue  # No DE in the very first stage

                    if not self.subpopulations.get(budget):
                        # iter-1; bracket-1; stage-2;3;4... -> vanilla SH
                        if s == self.s_max:
                            if prev_budget in self.subpopulations:
                                sorted_prev = sorted(self.history, key=lambda x: x[1], reverse=not self.is_minimize)
                                selected = [cfg for cfg, _, b in sorted_prev if b == prev_budget][:n_i]
                                configs = [self.decode_vector(vec) for vec in selected]
                                results, _ = self.evaluate_configs(configs, factors)
                                vectors = [self.encode_config(cfg) for cfg, _, _ in results]
                                self.subpopulations[budget] = vectors
                                self.global_pool.setdefault(budget, []).extend(vectors)
                                for (cfg, perf, _), vec in zip(results, vectors):
                                    self.history.append((vec, perf, budget))
                                if budget == self.R * self.R_unit:
                                    self.hf_evaluated_configs.extend(results)
                                continue  # no DE at vanilla SH

                    new_pop = []
                    for j in range(n_i):
                        # [iter-1; bracket >=2] or [iter >=2] --> retrieve subpop according the same budget level
                        # rolling pointer for DE [budget](select target from previous bracket/iter; ➡️ )
                        target = self.subpopulations[budget][j % len(self.subpopulations[budget])]
                        if i == 0:
                            # Vanilla DE: parent_pool = subpopulation [the first stage for all bracket]
                            parent_pool = self.subpopulations[budget]
                        else:
                            # Modified DE: parent_pool = self.subpopulations.get(prev_budget, []); ⬇️
                            parent_pool = self.subpopulations.get(prev_budget, [])

                        mutant = self.mutate(parent_pool.copy(), budget)
                        trial = self.crossover(target, mutant)
                        decoded_trial = self.decode_vector(trial)
                        results, _ = self.evaluate_configs([decoded_trial], factors)
                        perf, evaluated_cost = results[0][1], results[0][2]

                        self.global_pool.setdefault(budget, []).append(trial)
                        self.history.append((trial, perf, budget))

                        # Compare trial vs target; match the perf of target, allclose can fault-tolerant comparison,
                        # tolerating some loss of precision due to normalization operations.
                        target_perf = next((p for v, p, b in self.history if np.allclose(v, target) and b == budget), None)
                        if target_perf is None or \
                            (self.is_minimize and perf < target_perf) or \
                            (not self.is_minimize and perf > target_perf):
                            new_pop.append(trial)
                        else:
                            new_pop.append(target)

                        if budget == self.R * self.R_unit:
                            self.hf_evaluated_configs.append((decoded_trial, perf, evaluated_cost))

                        if self.consumed_cost >= self.total_budget:
                            break

                    self.subpopulations[budget] = new_pop
                    if self.consumed_cost >= self.total_budget:
                        break

            iteration += 1

        self.pbar.close()
        end_time = time.time()
        runtime = end_time - start_time
        print(f"[DONE] Total Cost: {self.consumed_cost:.2f} | Time: {runtime:.2f}s")
        self.logger.store_runtime_to_csv(runtime, self.log_path)


    def random_sample_one(self):
        config = {}
        for key, val in self.knobs_info.items():
            if val['type'] == 'integer':
                config[key] = random.randint(val['min'], val['max'])
            elif val['type'] == 'float':
                config[key] = random.uniform(val['min'], val['max'])
            elif val['type'] == 'enum':
                config[key] = random.choice(val['enum_values'])
        return config

    def encode_config(self, config):
        vector = []
        for key in self.knobs_info:
            info = self.knobs_info[key]
            val = config[key]
            if info['type'] == 'enum':
                idx = info['enum_values'].index(val)
                normalized = idx / (len(info['enum_values']) - 1)
            else:
                normalized = (val - info['min']) / (info['max'] - info['min'])
            vector.append(normalized)
        return np.array(vector)

    def decode_vector(self, vector):
        config = {}
        for i, key in enumerate(self.knobs_info):
            info = self.knobs_info[key]
            x = vector[i]
            if info['type'] == 'enum':
                idx = int(round(x * (len(info['enum_values']) - 1)))
                idx = max(0, min(idx, len(info['enum_values']) - 1))
                config[key] = info['enum_values'][idx]
            elif info['type'] == 'integer':
                val = int(round(info['min'] + x * (info['max'] - info['min'])))
                config[key] = val
            else:
                val = info['min'] + x * (info['max'] - info['min'])
                config[key] = val
        return config

    # def mutate(self, pool, budget):
    #     if len(pool) < 3:
    #         #  global pop pool is used for edge cases.
    #         pool += random.sample(self.global_pool.get(budget, []), 3 - len(pool))
    #     r1, r2, r3 = random.sample(pool, 3)
    #     mutant = r1 + self.f * (r2 - r3)
    #     return np.clip(mutant, 0.0, 1.0)

    def mutate(self, pool, budget):
        if len(pool) < 3:
            pool += random.sample(self.global_pool.get(budget, []), 3 - len(pool))
        r1, r2, r3 = random.sample(pool, 3)
        mutant = r1 + self.f * (r2 - r3)

        # Fix out-of-bounds values, random sampling when out of bounds
        for i in range(len(mutant)):
            if mutant[i] < 0.0 or mutant[i] > 1.0:
                mutant[i] = random.uniform(0.0, 1.0)
        return mutant

    def crossover(self, target, mutant):
        dim = len(target)
        trial = np.copy(target)
        jrand = random.randint(0, dim - 1)
        for j in range(dim):
            if random.random() < self.cr or j == jrand:
                trial[j] = mutant[j]
        return trial

    def evaluate_configs(self, configs, factors, stage_budget=None):
        evaluated_configs = []
        cost_configs = 0
        current_loop_consumption = 0
        # restart container to avoid fresh environment.
        self.target_system.restart_container()
        for config in configs:

            # Evaluate workload and record performance & cost
            num_iter = 1
            total_perf = total_cost = 0
            for _ in range(num_iter):
                start = time.time()
                try:
                    run_time, compile_time, exe_size, csmith_time, source_size, source_lines = self.workload_controller.run_workload(
                        factors, config)
                except Exception as e:
                    print(f"Workload execution failed: {e}")
                    self.target_system.restart_container()
                    time.sleep(5)
                    run_time = compile_time = exe_size = csmith_time = source_size = source_lines = 999999999
                end = time.time()
                duration = end - start

                if self.optimize_objective == 'compile_time':
                    total_perf += compile_time
                elif self.optimize_objective == 'run_time':
                    total_perf += run_time
                total_cost += duration

            avg_perf = total_perf / num_iter
            avg_cost = total_cost / num_iter

            self.consumed_cost += avg_cost
            cost_configs += avg_cost
            current_loop_consumption += avg_cost
            evaluated_configs.append((config, avg_perf, avg_cost))

            # record data
            metrics = {
                self.optimize_objective: avg_perf,
                'compile_time': compile_time,
                'exe_size': exe_size,
                'csmith_time': csmith_time,
                'source_size': source_size,
                'LOC': source_lines,
                'cost': avg_cost
            }

            self.logger.logging_data2(config, metrics, factors, self.log_path, self.log_file)
            # self.logger.logging_data(config, avg_perf, avg_cost, factors, self.log_path, self.log_file)
            self.logger.logging_cyber_twin(config, avg_perf, avg_cost, factors, self.cyber_twin_path,
                                           self.cyber_twin_file)

            if self.consumed_cost >= self.total_budget:
                print(f"Total budget exhausted: {self.consumed_cost}")

                break
            if stage_budget is not None and current_loop_consumption >= stage_budget:
                print("stage budge exhausted.")
                break

        return evaluated_configs, cost_configs
