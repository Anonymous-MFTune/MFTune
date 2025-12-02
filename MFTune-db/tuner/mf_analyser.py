import csv
import os
import time
from systems.mysqldb import MysqlDB
from systems.postgresqldb import PostgresqlDB
from workload import WorkloadController
from utils.multfidelity_optimizer import MultiFidelityOptimizer
from utils.logger import Logger


class MFAnalyser:
    def __init__(self, args_db, args_workload, args_tune, run):
        super(MFAnalyser, self).__init__()
        self.args_db = args_db
        self.args_workload = args_workload
        self.args_tune = args_tune
        self.run = run


        self.workload_bench = args_workload["workload_bench"]
        self.fidelity_type = args_tune['fidelity_type']
        self.fidelity_metric = args_tune['fidelity_metric']
        self.optimize_objective = args_tune['optimize_objective']
        self.sys_name = self.args_db['db']
        self.tuning_method = args_tune['tuning_method']

        # retrieve low-fidelity configs from log_path
        self.log_path = f'experimental_results/{self.sys_name}/{self.workload_bench}/{self.tuning_method}/run_{run}_ga_{self.fidelity_type}'


        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'
        self.max_iter = int(args_tune['max_iter'])


        if self.sys_name == 'mysql':
            self.target_system = MysqlDB(self.args_db)
        elif self.sys_name == 'postgresql':
            self.target_system = PostgresqlDB(self.args_db)

        self.workload_controller = WorkloadController(args_db, args_workload, self.target_system)

        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)

        self.multi_fidelity_optimizer = MultiFidelityOptimizer(self.workload_controller, self.target_system.knobs_info,
                                                               self.evaluate_config_pop, self.target_system,
                                                               self.max_iter, self.fidelity_metric, self.logger)

    def analyse_and_verify_configs(self):
        hf_factors = self.workload_controller.get_default_fidelity_factors()
        for file_name in os.listdir(self.log_path):
            if file_name.startswith('lf') and file_name.endswith('.csv') and 'on_hf' not in file_name:
                config_pop = self.read_configs_from_csv(file_name)
                evaluated_config_pop, _ = self.evaluate_config_pop(config_pop, hf_factors)

                new_file_name = file_name.replace('.csv', '_on_hf.csv')
                self.logger.verified_config_pop_to_csv(evaluated_config_pop, hf_factors, 1, self.log_path, new_file_name)

    def read_configs_from_csv(self, file_name):
        file_path = os.path.join(self.log_path, file_name)
        config_pop = []
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                config = {key: value for key, value in row.items() if key in self.target_system.knobs_info}
                config_pop.append(config)
        return config_pop

    def evaluate_config_pop(self, config_pop, factors):

        evaluated_config_pop = []
        cost_config_pop = 0

        for config in config_pop:
            print(f"Set the config to {self.sys_name}: {config}")
            # Retrieve data from Cyber-Twin
            cached_result = self.retrieve_from_cyber_twin(config, factors)
            if cached_result:
                perf, evaluated_cost, prepare_time, run_time, clean_time = cached_result
            else:

                self.target_system.set_db_knob(config)
                # Remark: increasing the num_iter can improve the robustness/reliability  of measurement.
                num_iter = 1
                total_perf = 0
                total_prepare_time = 0
                total_run_time = 0
                total_clean_time = 0
                total_evaluated_cost = 0
                for _ in range(num_iter):
                    start = time.time()
                    try:
                        print(f"Execute workload: {factors}")
                        latency, throughput, prepare_time, run_time, clean_time = self.workload_controller.run_workload(
                            factors)
                    except Exception as e:
                        print(f"Workload execution failed: {e}")
                        latency = throughput = prepare_time = run_time = clean_time = 0
                    end = time.time()
                    evaluated_cost = end - start
                    # accumulative perf result
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

                print(f"[PERFORMANCE]: {self.optimize_objective}: {perf}")
                print("-------------------------------------------------")

            evaluated_config_pop.append((config, perf, evaluated_cost))
            self.logger.logging_cyber_twin(config, perf, evaluated_cost, factors, prepare_time, run_time, clean_time,
                                           self.cyber_twin_path, self.cyber_twin_file)

        return evaluated_config_pop, cost_config_pop

    def retrieve_from_cyber_twin(self, config, fidelity):
        """Retrieve from Cyber-Twin.csv """
        cyber_twin_file = os.path.join(self.cyber_twin_path, self.cyber_twin_file)
        if not os.path.exists(cyber_twin_file):
            return None

        with open(cyber_twin_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)

            # obtain index
            num_knobs = len(self.target_system.knobs_info)
            perf_index = num_knobs
            evaluated_cost_index = num_knobs + 1
            fidelity_index = num_knobs + 2
            prepare_time_index = num_knobs + 3
            run_time_index = num_knobs + 4
            clean_time_index = num_knobs + 5

            for row in reader:
                stored_config = {header[i]: row[i] for i in range(num_knobs)}
                stored_fidelity = eval(row[fidelity_index])

                if stored_config == config and stored_fidelity == list(fidelity.values()):
                    perf = float(row[perf_index])
                    evaluated_cost = float(row[evaluated_cost_index])
                    prepare_time = float(row[prepare_time_index])
                    run_time = float(row[run_time_index])
                    clean_time = float(row[clean_time_index])
                    return (perf, evaluated_cost, prepare_time, run_time, clean_time)

        return None

