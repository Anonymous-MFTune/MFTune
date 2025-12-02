import math
import random
import time
from pyDOE import lhs

from systems.gcc_compiler import GccCompiler
from systems.clang_compiler import ClangCompiler
from utils.server_connector import ServerConnector
from workload import WorkloadController
from utils.multfidelity_optimizer import MultiFidelityOptimizer
from utils.logger import Logger
import sys
from tqdm import tqdm
from utils.config_utils import ConfigUtils


class MF2Tuner:
    def __init__(self, args_compiler, args_workload, args_tune, run):
        super(MF2Tuner, self).__init__()
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
        self.log_file = 'GATuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'
        self.budget_for_fidelity_search = 0  # distribute a certain budget when using multi fidelity optimization

        # Parameters Settings for Algorithm: GA for config space
        self.config_pop_size = 20
        self.mutation_rate = 1/int(self.args_compiler['knob_num'])
        self.crossover_rate = 0.9
        self.evaluated_configs = set()
        self.consumed_cost = 0

        # Target system
        if self.sys_name == 'gcc':
            self.target_system = GccCompiler(args_compiler)
        elif self.sys_name == 'clang':
            self.target_system = ClangCompiler(args_compiler)

        # Controller; Logger; MF_Optimizer
        self.workload_controller = WorkloadController(args_compiler, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)
        self.multi_fidelity_optimizer = MultiFidelityOptimizer(self.workload_controller, self.workload_controller.fidelity_factors_info,
                                                               self.evaluate_config_pop, self.target_system, self.optimize_objective,
                                                               self.max_iter, self.fidelity_metric)

        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")

        # Settings for multi-fidelity
        self.hf_evaluated_configs = []  # Record configs that have been evaluated under the high-fidelity setting
        self.PROB_HIGH_FIDELITY_TRIGGER = 0.5  # Probability for triggering high-fidelity evaluation
        self.sample_size_4_fidelity_quantification = 10  # Sample size for fidelity quantification
        self.hf_factors = self.workload_controller.get_default_fidelity_factors()
        self.rnd_factors = self.workload_controller.get_random_fidelity_factors()  # Random fidelity setting
        self.budget_4_fidelity_search = self.total_budget * 0.25  # Budget for fidelity search

    def tune_mf2(self):
        start_time = time.time()

        if self.fidelity_type == 'single_fidelity':
            self.evolutionary_search_config(self.total_budget, self.hf_factors)

        elif self.fidelity_type == 'multi_fidelity':

            print("=========== FIDELITY SPACE ANALYSIS ===========")
            selected_fidelities = self.fidelity_optimization_and_selection()
            if selected_fidelities is None:
                print(f"Without suitable fidelity setting, using high-fidelity search: {self.total_budget - self.consumed_cost}")
                self.evolutionary_search_config(self.total_budget - self.consumed_cost, self.hf_factors)
                end_time = time.time()
                runtime = end_time - start_time
                self.logger.store_runtime_to_csv(runtime, self.log_path)
                return
            lf_factors, lf_corr, lf_cost = selected_fidelities[0]  # Only use one low-fidelity setting
            print(f"FIDELITY SETTING: {lf_factors} | CORRELATION: {lf_corr} | COST: {lf_cost}")
            print(f"[DEBUG] Predefined Budget for Fidelity Optimization: {self.budget_4_fidelity_search}| Actual Budget Consumed: {self.consumed_cost}")

            print("=========== CONFIGURATION SPACE ANALYSIS ==========")
            budget_4_hfea = self.total_budget * 0.5  # preserve 30% budget for the last high-fidelity EA
            if self.total_budget - self.consumed_cost < budget_4_hfea:
                print(f"There is not enough budget for hfea, skip lf optimization and conduct hf optimization.")
                self.evolutionary_search_config(self.total_budget - self.consumed_cost, self.hf_factors)
                return


            fidelity_id = 1

            print(f"[2-LFEA] Evolutionary Search Under Low-Fidelity Using Promising Configs Filtered by LHS as Initial Pop")
            budget_4_lfea = self.total_budget - self.consumed_cost - budget_4_hfea
            fidelity_id += 1
            lfea_evaluated_configs = self.evolutionary_search_config(budget_4_lfea, lf_factors, None,
                                                                     lf_corr, fidelity_id)
            promising_configs = [config for config, _, _ in lfea_evaluated_configs[:self.config_pop_size]]
            print(f"[DEBUG] Predefined Budget for LFEA: {budget_4_lfea} | Fidelity Id: {fidelity_id}")
            print(f"[DEBUG] The Remaining Budget: {self.total_budget - self.consumed_cost}")

            print(f"[3-HFEA] Evolutionary Search Under High-Fidelity Using Promising Configs Evolved by LFEA as Initial Pop")
            budget_4_hfea = self.total_budget - self.consumed_cost
            print(f"Budget for HFEA: {budget_4_hfea}")
            self.evolutionary_search_config(budget_4_hfea, self.hf_factors,
                                            init_config_pop=promising_configs,
                                            hf_evaluated_configs=self.hf_evaluated_configs)

        end_time = time.time()
        runtime = end_time - start_time
        self.logger.store_runtime_to_csv(runtime, self.log_path)
        self.pbar.close()


    @staticmethod
    def calculate_centroid(cluster):
        """
        calculate the center of the cluster。
        :param cluster: cluster with multiple solutions
        :return: the fidelity of center
        """
        return sum(ind[1] for ind in cluster) / len(cluster)

    def evolutionary_search_config(self, budget, fidelity, init_config_pop=None, kd_corr=1, fidelity_id=0,
                                   hf_evaluated_configs=None):

        """
        :param hf_evaluated_configs:
        :param budget: the budget for evolutionary search
        :param fidelity: the fidelity for evolutionary search
        :param init_config_pop: initial population, which could be empty or the population optimized under the other fidelity settings
        :param kd_corr: fidelity value (default as 1) for file naming, is used to record the results of different fidelity
        :param fidelity_id: fidelity id, default as 0 (denotes as full/high-fidelity)
        :return:
        """

        generation = 0
        consumed_cost = 0
        if init_config_pop is None:
            init_config_pop = self.initialize_config_pop()

        # make sure the size of init_config_pop is at least greater than or equal to pop_size
        if len(init_config_pop) <= self.config_pop_size:
            init_config_pop = self.supplement_population(init_config_pop)

        evaluated_config_pop, cost_init_config_pop = self.evaluate_config_pop(init_config_pop, fidelity)
        consumed_cost += cost_init_config_pop

        # if there exist configs (filtered by lf and evaluated under high-fidelity), combine them with initial-config_pop
        if hf_evaluated_configs:
            evaluated_config_pop = evaluated_config_pop + hf_evaluated_configs

        if self.optimize_objective in ['throughput', 'RPS']:
            evaluated_config_pop.sort(key=lambda x: x[1], reverse=True)
        elif self.optimize_objective in ['latency', 'TPR', 'run_time']:
            evaluated_config_pop.sort(key=lambda x: x[1])

        evaluated_config_pop = evaluated_config_pop[:self.config_pop_size]

        # Record the evaluated configs / with fidelity as part of the key
        for config, _, _ in evaluated_config_pop:
            self.evaluated_configs.add((tuple(sorted(config.items())), tuple(sorted(fidelity.items()))))

        self.logger.store_config_pop_to_csv(evaluated_config_pop, generation, fidelity, kd_corr, fidelity_id,
                                            self.log_path)

        while consumed_cost < budget:
            generation += 1
            config_offspring = []
            while len(config_offspring) < self.config_pop_size:
                parent1, parent2 = self.select_config_parents(evaluated_config_pop)
                child1, child2 = self.uniform_crossover_config(parent1, parent2)
                child1 = self.mutate_config(child1)
                child2 = self.mutate_config(child2)

                # Check and make sure that the offspring is unique and unevaluated
                child1_tuple = (tuple(sorted(child1.items())), tuple(sorted(fidelity.items())))
                child2_tuple = (tuple(sorted(child2.items())), tuple(sorted(fidelity.items())))
                if child1_tuple not in self.evaluated_configs and child1 not in config_offspring:
                    config_offspring.append(child1)
                if len(config_offspring) < self.config_pop_size and child2_tuple not in self.evaluated_configs and child2 not in config_offspring:
                    config_offspring.append(child2)

            # Evaluate the offspring
            evaluated_config_offspring, cost_config_offspring = self.evaluate_config_pop(config_offspring, fidelity)
            consumed_cost += cost_config_offspring

            # Record the evaluated offspring configs / with fidelity as part of the key
            for config, _, _ in evaluated_config_offspring:
                self.evaluated_configs.add((tuple(sorted(config.items())), tuple(sorted(fidelity.items()))))

            combined_config_pop = evaluated_config_pop + evaluated_config_offspring

            # Dynamically adjust the sorting logic according to the optimization objective
            if self.optimize_objective in ['throughput', 'RPS']:
                combined_config_pop.sort(key=lambda x: x[1], reverse=True)
            elif self.optimize_objective in ['latency', 'TPR', 'run_time']:
                combined_config_pop.sort(key=lambda x: x[1])

            evaluated_config_pop = combined_config_pop[:self.config_pop_size]
            self.logger.store_config_pop_to_csv(evaluated_config_pop, generation, fidelity, kd_corr, fidelity_id,
                                                self.log_path)

            if fidelity_id != 0:
                # Trigger High-fidelity Evaluation (for the best config in current states)
                cost_best_config = self.trigger_hf_evaluation(evaluated_config_pop)
                consumed_cost += cost_best_config


        return evaluated_config_pop

    def initialize_config_pop(self):
        """initialize a config pop; make ensure no repeat configs are generated"""
        config_pop = []
        seen_configs = set()
        while len(config_pop) < self.config_pop_size:
            config = {}
            for knob_name, knob_info in self.target_system.knobs_info.items():
                if knob_info['type'] == 'integer':
                    random_value = random.randint(knob_info['min'], knob_info['max'])
                    config[knob_name] = random_value
                elif knob_info['type'] == 'float':
                    random_value = random.uniform(knob_info['min'], knob_info['max'])
                    config[knob_name] = random_value
                elif knob_info['type'] == 'enum':
                    possible_value = knob_info['enum_values']
                    index = random.randint(0, len(possible_value) - 1)
                    config[knob_name] = possible_value[index]
            config_tuple = tuple(config.items())
            if config_tuple not in seen_configs:
                config_pop.append(config)
                seen_configs.add(config_tuple)

        return config_pop

    def evaluate_config_pop(self, configs, factors, stage_budget=None):
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
                    print(f"Execute workload: {factors}")
                    run_time, compile_time, exe_size, csmith_time, source_size, source_lines = self.workload_controller.run_workload(factors, config)

                except Exception as e:
                    print(f"Error during workload execution: {e}")
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

    def select_config_parents(self, evaluated_config_pop):
        """Binary tournament selection"""

        candidate1, candidate2 = random.sample(evaluated_config_pop, 2)
        candidate3, candidate4 = random.sample(evaluated_config_pop, 2)

        if self.optimize_objective in ['throughput', 'RPS']:
            parent1 = candidate1[0] if candidate1[1] > candidate2[1] else candidate2[0]
            parent2 = candidate3[0] if candidate3[1] > candidate4[1] else candidate4[0]
        elif self.optimize_objective in ['latency', 'run_time']:
            parent1 = candidate1[0] if candidate1[1] < candidate2[1] else candidate2[0]
            parent2 = candidate3[0] if candidate3[1] < candidate4[1] else candidate4[0]
        return parent1, parent2

    def single_point_crossover_config(self, parent1, parent2):
        """single point crossover"""

        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # identify the position for crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = {}
        child2 = {}

        # obtain the names of all knobs
        knob_names = list(parent1.keys())

        for i in range(len(knob_names)):
            if i < crossover_point:
                child1[knob_names[i]] = parent1[knob_names[i]]
                child2[knob_names[i]] = parent2[knob_names[i]]
            else:
                child1[knob_names[i]] = parent2[knob_names[i]]
                child2[knob_names[i]] = parent1[knob_names[i]]
        return child1, child2

    def uniform_crossover_config(self, parent1, parent2):
        """uniform crossover"""
        # Decide whether to crossover or not according to the probability
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = {}
        child2 = {}

        # Get the name of all knobs
        knob_names = list(parent1.keys())

        # For each knob, uniformly choose whether to inherit from parent1 or parent2
        for knob in knob_names:
            if random.random() < 0.5:
                child1[knob] = parent1[knob]
                child2[knob] = parent2[knob]
            else:
                child1[knob] = parent2[knob]
                child2[knob] = parent1[knob]

        return child1, child2

    def mutate_config(self, config):
        """Mutation"""
        for knob in config.keys():
            if random.random() < self.mutation_rate:
                knob_info = self.target_system.knobs_info[knob]
                if knob_info['type'] == 'integer':
                    config[knob] = random.randint(knob_info['min'], knob_info['max'])
                elif knob_info['type'] == 'float':
                    config[knob] = random.uniform(knob_info['min'], knob_info['max'])
                elif knob_info['type'] == 'enum':
                    config[knob] = random.choice(knob_info['enum_values'])
        return config

    def sampling_configs_by_lhs(self, sample_size):
        num_params = len(self.target_system.knobs_info)
        lhs_sample = lhs(num_params, samples=sample_size)
        configs = []

        for i in range(sample_size):
            config = {}
            for j, (key, val) in enumerate(self.target_system.knobs_info.items()):
                if val['type'] == 'integer':
                    range_width = val['max'] - val['min'] + 1
                    scaled_value = int(lhs_sample[i][j] * range_width) + val['min']
                    config[key] = scaled_value
                elif val['type'] == 'float':
                    range_width = val['max'] - val['min']
                    scaled_value = lhs_sample[i][j] * range_width + val['min']
                    config[key] = scaled_value
                elif val['type'] == 'enum':
                    possible_values = val['enum_values']
                    index = int(lhs_sample[i][j] * len(possible_values))
                    config[key] = possible_values[index]
            configs.append(config)

        return configs


    def supplement_population(self, init_pop):
        additional_configs = self.initialize_config_pop()
        existing_configs_set = set(tuple(sorted(c.items())) for c in init_pop)
        for config in additional_configs:
            if tuple(sorted(config.items())) not in existing_configs_set:
                init_pop.append(config)
                if len(init_pop) >= self.config_pop_size:
                    break
        return init_pop


    def fidelity_optimization_and_selection(self):

        config_samples = ConfigUtils.sampling_configs_by_lhs(self.sample_size_4_fidelity_quantification,
                                                             self.target_system.knobs_info)

        hf_evaluated_samples, hf_evaluated_samples_cost = self.evaluate_config_pop(config_samples, self.hf_factors)
        self.hf_evaluated_configs.extend(hf_evaluated_samples)
        hf_perf = [perf for _, perf, _ in hf_evaluated_samples]
        hf_cost = [cost for _, _, cost in hf_evaluated_samples]

        cost_related_factors, cost_dva = self.multi_fidelity_optimizer.decision_variable_analysis(
            hf_evaluated_samples, self.hf_factors, 5, 5)
        self.logger.log_cost_related_factors(cost_related_factors, self.log_path)

        if not cost_related_factors:
            print("No cost-related factors found, proceeding with single fidelity optimization.")
            self.evolutionary_search_config(self.total_budget - self.consumed_cost, self.hf_factors)
            return

        # Fidelity Optimization and Fidelity Measurement
        optimized_fidelity_pop = self.multi_fidelity_optimizer.evolutionary_search_fidelity(self.hf_factors,
                                                                                            cost_related_factors,
                                                                                            config_samples, hf_perf,
                                                                                            self.log_path, cost_dva,
                                                                                            self.budget_4_fidelity_search - self.consumed_cost)

        selected_fidelities = self.multi_fidelity_optimizer.select_fidelity_by_knee_point(optimized_fidelity_pop,
                                                                                          self.log_path)

        return selected_fidelities

    def trigger_hf_evaluation(self, lf_evaluated_configs):
        # Triggering High-Fidelity Evaluation (for the best config under current states)
        consumed_cost = 0
        current_best_config = ConfigUtils.get_top_k_configs(lf_evaluated_configs, 1,
                                                            self.optimize_objective)
        current_config_tuple = tuple(sorted(current_best_config[0].items()))
        evaluated_set = set(tuple(sorted(config.items())) for config, _, _ in self.hf_evaluated_configs)
        if current_config_tuple not in evaluated_set:
            hf_evaluated_current_best_config, consumed_cost = self.evaluate_config_pop(current_best_config,
                                                                                       self.hf_factors)
            self.hf_evaluated_configs.extend(hf_evaluated_current_best_config)

        return consumed_cost


