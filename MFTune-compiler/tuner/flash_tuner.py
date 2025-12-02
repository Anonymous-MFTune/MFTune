import random
import time

from systems.gcc_compiler import GccCompiler
from systems.clang_compiler import ClangCompiler
from workload import WorkloadController
from tqdm import tqdm
from utils.logger import Logger
from sklearn.tree import DecisionTreeRegressor
from utils.config_utils import ConfigUtils



class FLASHTuner:
    def __init__(self, args_compiler, args_workload, args_tune, run):
        super(FLASHTuner, self).__init__()
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
        self.log_file = 'FLASHTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        # Parameters Settings for Algorithm: Flash for config space
        self.initial_size = 30
        self.sampling_size = 1000
        self.evaluated_configs = set()
        self.consumed_cost = 0

        # Target system
        if self.sys_name == 'gcc':
            self.target_system = GccCompiler(args_compiler)
        elif self.sys_name == 'clang':
            self.target_system = ClangCompiler(args_compiler)

        # Workload Controller; Logger;
        self.workload_controller = WorkloadController(args_compiler, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)

    def tune_flash(self):
        start_time = time.time()

        if self.fidelity_type == 'single_fidelity':
            hf_factors = self.workload_controller.get_default_fidelity_factors()
            self.flash_search_config(self.total_budget, hf_factors)

        elif self.fidelity_type == 'multi_fidelity':
            pass

        end_time = time.time()
        runtime = end_time - start_time

        # Record runtime using the Logger class
        self.logger.store_runtime_to_csv(runtime, self.log_path)

    def flash_search_config(self, budget, fidelity, init_configs=None, kd_corr=1, fidelity_id=0,
                            evaluated_filtered_configs=None):
        """
        :param budget:
        :param fidelity:
        :param init_configs:
        :param kd_corr:
        :param fidelity_id:
        :param evaluated_filtered_configs:
        :return:
        """

        print("Start running flash...")
        pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")
        print("...")

        consumed_cost = 0
        if init_configs is None:
            # init_configs = self.sampling_configs(self.initial_size)
            init_configs = ConfigUtils.sampling_configs_by_lhs(self.initial_size, self.target_system.knobs_info)
        evaluated_configs, cost_init_configs = self.evaluate_configs(init_configs, fidelity)
        consumed_cost += cost_init_configs
        pbar.update(cost_init_configs)

        # prepared for extension to multi-fidelity
        if evaluated_filtered_configs:
            # Combine and sort the results from both evaluations
            evaluated_configs = evaluated_configs + evaluated_filtered_configs
            if self.optimize_objective == '---':
                evaluated_configs.sort(key=lambda x: x[1], reverse=True)
            elif self.optimize_objective == 'run_time':
                evaluated_configs.sort(key=lambda x: x[1])

        evaluated_configs = evaluated_configs[:self.initial_size]
        # Record the evaluated configs / with fidelity as part of the key
        for config, _, _ in evaluated_configs:
            self.evaluated_configs.add((tuple(sorted(config.items())), tuple(sorted(fidelity.items()))))

        while consumed_cost < budget:

            # Train a CART: regression decision tree model
            model = DecisionTreeRegressor()
            configs = [config for config, _, _ in evaluated_configs]
            train_x = self.preprocess_configs_with_knobs_info(configs, self.target_system.knobs_info)
            # train_x = [list(config.values()) for config, _, _ in evaluated_configs]
            train_y = [perf for _, perf, _ in evaluated_configs]
            model.fit(train_x, train_y)

            # sampled_configs = self.sampling_configs(self.sampling_size)
            sampled_configs = ConfigUtils.sampling_configs_by_rs(self.sampling_size, self.target_system.knobs_info)
            # Filter those configs that haven't been evaluated in the past
            unevaluated_configs = [config for config in sampled_configs if (
                tuple(sorted(config.items())), tuple(sorted(fidelity.items()))
            ) not in self.evaluated_configs]

            # Transform the unevaluated configs at two-dimensional list, get the value
            # As some of the configs that may take values as strings, preprocess is needed
            test_x = self.preprocess_configs_with_knobs_info(unevaluated_configs, self.target_system.knobs_info)
            # test_x = [list(config.values()) for config in unevaluated_configs]

            # Predict the performance for given configs by trained model
            predicted_performances = model.predict(test_x)

            if self.optimize_objective == '---':
                best_config_index = predicted_performances.argmax()
            elif self.optimize_objective == 'run_time':
                best_config_index = predicted_performances.argmin()
            else:
                raise ValueError(f"Unsupported optimization objective: {self.optimize_objective}")

            best_config = unevaluated_configs[best_config_index]

            # Implement true system measurement for estimated best config
            evaluated_best_config, cost_best_config = self.evaluate_configs([best_config], fidelity)
            consumed_cost += cost_best_config
            pbar.update(cost_best_config)

            # Update the evaluated config (training data)
            evaluated_configs += evaluated_best_config
            self.evaluated_configs.add((tuple(sorted(best_config.items())), tuple(sorted(fidelity.items()))))
        pbar.close()

        return evaluated_configs

    def sampling_configs(self, sampling_size):
        """Random sampling"""
        init_configs = []
        seen_configs = set()
        while len(init_configs) < sampling_size:
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
                init_configs.append(config)
                seen_configs.add(config_tuple)

        return init_configs

    def preprocess_configs_with_knobs_info(self, configs, knobs_info):
        """
        Preprocessing configs dynamically according to knobs_info
        :param configs: config list，dict{config_name: config_value}
        :param knobs_info: detailed info for config
        :return: transformed two-dimensional list
        """
        processed_configs = []
        for config in configs:
            processed_config = []
            for key, value in config.items():
                if key in knobs_info and knobs_info[key]["type"] == "enum":
                    # extract the index as it's value for the purpose of training model
                    enum_values = knobs_info[key]["enum_values"]
                    processed_config.append(enum_values.index(value))
                else:
                    processed_config.append(value)
            processed_configs.append(processed_config)
        return processed_configs

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

