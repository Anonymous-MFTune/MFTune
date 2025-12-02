from openbox import Optimizer
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter

from systems.clang_compiler import ClangCompiler
from systems.gcc_compiler import GccCompiler
from utils.server_connector import ServerConnector
from workload import WorkloadController
import time
from utils.logger import Logger



class SMACTuner:
    def __init__(self, args_compiler, args_workload, args_tune, run):
        super(SMACTuner, self).__init__()
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
        self.log_file = 'SMACTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'
        self.run = run

        # Target system
        if self.sys_name == 'gcc':
            self.target_system = GccCompiler(args_compiler)
        elif self.sys_name == 'clang':
            self.target_system = ClangCompiler(args_compiler)

        # Parameters Settings for Algorithm:
        self.evaluated_configs = set()
        self.consumed_cost = 0
        self.config_space = self.get_configspace()

        # Workload Controller; Logger;
        self.workload_controller = WorkloadController(args_compiler, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)


    def get_configspace(self):
        """
        Define the config space
        """
        cs = ConfigurationSpace()
        for knob_name, knob_info in self.target_system.knobs_info.items():
            if knob_info['type'] == 'integer':
                cs.add_hyperparameter(UniformIntegerHyperparameter(knob_name, knob_info['min'], knob_info['max']))
            elif knob_info['type'] == 'float':
                cs.add_hyperparameter(UniformFloatHyperparameter(knob_name, knob_info['min'], knob_info['max']))
            elif knob_info['type'] == 'enum':
                cs.add_hyperparameter(CategoricalHyperparameter(knob_name, knob_info['enum_values']))
        return cs

    def objective_function(self, config):
        """
        Define objective function: call the evaluate_configs to evaluate a given config
        """
        config_dict = config.get_dictionary()
        factors = self.workload_controller.get_default_fidelity_factors()

        if self.consumed_cost >= self.total_budget:
            print("Time limit exceeded. Skipping evaluation")
            return {"objectives": [0]}

        # Call evaluate_configs to evaluate a given config under the target system and workload
        evaluated_configs, cost = self.evaluate_configs([config_dict], factors)
        performance = evaluated_configs[0][1]

        # OpenBox implements the minimization objective by default, which should be transformed if we expect to maximize
        result = -performance if self.optimize_objective == 'RPS' else performance

        # The idea format of objectives for OpenBox
        return {"objectives": [result]}


    def tune_smac(self):
        """
        Config Tuning by using SMAC within OpenBox framework
        """
        start_time = time.time()
        print("Start running SMAC...")


        # Note: to embed openbox into our framework seamlessly, we pass max_runs instead of max_runtime.
        # But it will also work, as we have limited the resource usage in our framework. max_runs should large enough.

        # Initialization
        optimizer = Optimizer(
            objective_function=self.objective_function,
            config_space=self.config_space,
            max_runs=self.max_iter,
            task_id=f"SMAC_Run_{self.run}",
            advisor_type='default',
            surrogate_type='prf',  # Random Forest
            acq_type='ei',
            init_strategy='random',
            initial_runs=30  # Initial seeds
        )

        history = optimizer.run()

        # Obtain all the evaluated configs and corresponding performances
        all_configs = history.get_config_dicts()
        all_objectives = history.get_objectives()

        # Get the best config and corresponding perf
        best_index = min(range(len(all_objectives)), key=lambda i: all_objectives[i])
        best_config = all_configs[best_index]
        best_perf = all_objectives[best_index]

        # If the optimization objective is RPS, we need to take the inverse to recover the original performance
        if self.optimize_objective == 'RPS':
            best_perf = -best_perf

        print(f"Best Configuration: {best_config}, Best Performance: {best_perf}")

        end_time = time.time()
        runtime = end_time - start_time

        # Record runtime using the Logger class
        self.logger.store_runtime_to_csv(runtime, self.log_path)

        return best_perf

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


