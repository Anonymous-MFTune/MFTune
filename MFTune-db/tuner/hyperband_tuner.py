import random
import time
from pyDOE import lhs
from systems.mysqldb import MysqlDB
from systems.postgresqldb import PostgresqlDB
from workload import WorkloadController
from tqdm import tqdm
from utils.logger import Logger
from math import log, ceil
from utils.config_utils import ConfigUtils


class HBTuner:
    def __init__(self, args_db, args_workload, args_tune, run):
        super(HBTuner, self).__init__()
        self.max_iter = int(args_tune['max_iter'])
        self.total_budget = int(args_tune['total_budget'])
        self.args_db = args_db
        self.args_workload = args_workload
        self.args_tune = args_tune
        self.workload_bench = args_workload["workload_bench"]
        self.tuning_method = args_tune['tuning_method']
        self.fidelity_type = args_tune['fidelity_type']
        self.fidelity_metric = args_tune['fidelity_metric']
        self.optimize_objective = args_tune['optimize_objective']
        self.sys_name = self.args_db['db']
        self.log_path = f'experimental_results/{self.sys_name}/{self.workload_bench}/{self.tuning_method}/run_{run}_{self.tuning_method}_{self.fidelity_type}'
        self.log_file = 'HBTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        # Target system
        if self.sys_name == 'mysql':
            self.target_system = MysqlDB(self.args_db)
        elif self.sys_name == 'postgresql':
            self.target_system = PostgresqlDB(self.args_db)

        # Workload Controller; Logger;
        self.workload_controller = WorkloadController(args_db, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)

        # Parameters Settings for Algorithm
        self.default_factors = self.workload_controller.get_default_fidelity_factors()
        self.R_unit = 10  # Define resource unit
        self.R = self.default_factors['time'] / self.R_unit  # Maximum resource unit per config (only time factor)
        self.eta = 2  # Define config down-sampling rate
        self.s_max = int(log(self.R) / log(self.eta))  # Maximum stage in HS, as well as maximum bracket
        self.B = (self.s_max + 1) * self.R  # budget per bracket, i.e., budget for each full HS process

        self.evaluated_configs = set()
        self.consumed_cost = 0

        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")

    def tune_hyperband(self):

        start_time = time.time()
        # Continuously run the HyperBand within the constraint of total budget
        iteration = 1
        while self.consumed_cost < self.total_budget:
            print(f"Executing the {iteration}-th iteration of hyperband ...")
            # outer loop (Bracket)
            for s in reversed(range(self.s_max + 1)):
                # Initial number of configs
                n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
                # Initial number of resource unit per config
                r = self.R * self.eta ** (-s)

                configs = ConfigUtils.sampling_configs_by_lhs(n, self.target_system.knobs_info)  # Initialize n configs uniformly

                # Inner loop (Successive Halving)
                for i in range(s + 1):
                    n_i = int(n * self.eta ** (-i))  # number of configs for current stage
                    if i == s:
                        r_i = int(self.R)  # Make sure the evaluation is full resource unit for the config in last stage
                    else:
                        r_i = int(r * self.eta ** i)  # resource unit per config for current stage

                    budget = r_i * self.R_unit
                    factors = self.default_factors.copy()
                    factors['time'] = budget  # Control fidelity
                    print(f"[Iteration: {iteration}] | [Bracket: {self.s_max - s + 1}] | [SH stage: {i + 1}] | [Configs: {n_i}] | [Budget: {budget}s]")

                    evaluated_configs, cost = self.evaluate_configs(configs, factors)

                    # Resource exhaustion
                    if self.consumed_cost >= self.total_budget:
                        print("resource exhaustion.")
                        break

                    # Select a number of (n/eta) the best configus for the next HS loop (next stage)
                    if self.optimize_objective == 'throughput':
                        evaluated_configs.sort(key=lambda x: x[1], reverse=True)
                    elif self.optimize_objective == 'latency':
                        evaluated_configs.sort(key=lambda x: x[1])

                    configs = [config for config, _, _ in evaluated_configs[: max(1, n_i // self.eta)]]

        print(f'total consume time budget: {self.consumed_cost}')
        end_time = time.time()
        runtime = end_time - start_time
        self.logger.store_runtime_to_csv(runtime, self.log_path)
        self.pbar.close()

    def evaluate_configs(self, configs, factors, stage_budget=None):
        evaluated_configs = []
        cost_configs = 0
        current_loop_consumption = 0
        for config in configs:
            self.target_system.set_db_knob(config)
            num_iter = 1
            total_perf = total_prepare_time = total_run_time = total_clean_time = total_evaluated_cost = 0
            for _ in range(num_iter):
                start = time.time()
                try:
                    print(f"Execute workload: {factors}")
                    latency, throughput, prepare_time, run_time, clean_time = self.workload_controller.run_workload(factors)
                except Exception as e:
                    print(f"Workload execution failed: {e}")
                    latency = throughput = prepare_time = run_time = clean_time = 0
                end = time.time()
                evaluated_cost = end - start
                if self.optimize_objective == 'throughput':
                    total_perf += throughput
                elif self.optimize_objective == 'latency':
                    total_perf += latency
                total_prepare_time += prepare_time
                total_run_time += run_time
                total_clean_time += clean_time
                total_evaluated_cost += evaluated_cost

            perf = total_perf / num_iter
            prepare_time = total_prepare_time / num_iter
            run_time = total_run_time / num_iter
            clean_time = total_clean_time / num_iter
            evaluated_cost = total_evaluated_cost / num_iter

            self.pbar.update(evaluated_cost)
            print()
            print(f"[PERFORMANCE]: {self.optimize_objective}: {perf}")
            print("-------------------------------------------------")

            self.consumed_cost += evaluated_cost
            current_loop_consumption += evaluated_cost
            cost_configs += evaluated_cost
            evaluated_configs.append((config, perf, evaluated_cost))

            self.logger.logging_data(config, perf, evaluated_cost, factors, prepare_time, run_time, clean_time,
                                     self.log_path, self.log_file)
            self.logger.logging_cyber_twin(config, perf, evaluated_cost, factors, prepare_time, run_time, clean_time,
                                           self.cyber_twin_path, self.cyber_twin_file)

            if self.consumed_cost >= self.total_budget:
                break
            if stage_budget is not None and current_loop_consumption >= stage_budget:
                print("stage budge exhausted.")
                break
        return evaluated_configs, cost_configs


