import random
import numpy as np
import time
from systems.gcc_compiler import GccCompiler
from systems.clang_compiler import ClangCompiler
from workload import WorkloadController
from tqdm import tqdm
from utils.logger import Logger
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace


class HEBOTuner:
    def __init__(self, args_compiler, args_workload, args_tune, run):
        super(HEBOTuner, self).__init__()
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
        self.log_path = f'experimental_results/{self.sys_name}/{self.workload_bench}/{self.tuning_method}/run_{run}_{self.tuning_method}_{self.fidelity_type}'
        self.log_file = 'HEBOTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'
        self.consumed_cost = 0
        self.evaluated_configs = set()

        # Target system
        if self.sys_name == 'gcc':
            self.target_system = GccCompiler(args_compiler)
        elif self.sys_name == 'clang':
            self.target_system = ClangCompiler(args_compiler)

        self.workload_controller = WorkloadController(args_compiler, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)
        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")

        self.search_space = self.create_hebo_search_space()

        self.optimizer = HEBO(space=self.search_space, rand_sample=30)

    def create_hebo_search_space(self):
        params = []
        for knob_name, knob_info in self.target_system.knobs_info.items():
            if knob_info['type'] == 'integer':
                params.append({'name': knob_name, 'type': 'int', 'lb': knob_info['min'], 'ub': knob_info['max']})
            elif knob_info['type'] == 'float':
                params.append({'name': knob_name, 'type': 'num', 'lb': knob_info['min'], 'ub': knob_info['max']})
            elif knob_info['type'] == 'enum':
                params.append({'name': knob_name, 'type': 'cat', 'categories': knob_info['enum_values']})
        return DesignSpace().parse(params)

    def tune_hebo(self):
        start_time = time.time()

        if self.fidelity_type == 'single_fidelity':
            hf_factors = self.workload_controller.get_default_fidelity_factors()
            self.hebo_search_config(self.total_budget, hf_factors)
        elif self.fidelity_type == 'multi_fidelity':
            pass

        end_time = time.time()
        runtime = end_time - start_time
        self.logger.store_runtime_to_csv(runtime, self.log_path)
        self.pbar.close()

    def hebo_search_config(self, budget, fidelity):
        consumed_cost = 0

        while consumed_cost < budget:
            rec = self.optimizer.suggest(n_suggestions=1)
            config_dict = rec.iloc[0].to_dict()

            # Check if already evaluated
            if (tuple(sorted(config_dict.items())), tuple(sorted(fidelity.items()))) in self.evaluated_configs:
                continue

            evaluated_config, cost = self.evaluate_configs([config_dict], fidelity)
            perf = evaluated_config[0][1]
            # HEBO optimizes the minimization problem by default
            if self.optimize_objective in ['throughput', 'RPS']:
                perf = -perf
            self.optimizer.observe(rec, np.array([[perf]]))

            self.evaluated_configs.add((tuple(sorted(config_dict.items())), tuple(sorted(fidelity.items()))))
            consumed_cost += cost

        return

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
