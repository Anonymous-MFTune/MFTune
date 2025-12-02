import csv
import os
import time
from systems.httpd_server import HttpdServer
from systems.tomcat_server import TomcatServer
from utils.server_connector import ServerConnector
from workload import WorkloadController
from utils.multfidelity_optimizer import MultiFidelityOptimizer
from utils.logger import Logger


class MFAnalyser:
    def __init__(self, args_server, args_workload, args_tune, run):
        super(MFAnalyser, self).__init__()

        self.args_server = args_server
        self.password = args_server['password']
        self.server_url = args_server['url']
        self.args_workload = args_workload
        self.args_tune = args_tune
        self.run = run


        self.workload_bench = args_workload["workload_bench"]
        self.fidelity_type = args_tune['fidelity_type']
        self.fidelity_metric = args_tune['fidelity_metric']
        self.optimize_objective = args_tune['optimize_objective']
        self.sys_name = self.args_server['server']
        self.tuning_method = args_tune['tuning_method']

        # retrieve low-fidelity configs from log_path
        self.log_path = f'experimental_results/{self.sys_name}/{self.workload_bench}/ga/run_{run}_ga_{self.fidelity_type}'


        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'
        self.max_iter = int(args_tune['max_iter'])

        # Target system
        if self.sys_name == 'tomcat':
            self.target_system = TomcatServer(args_server)
        elif self.sys_name == 'httpd':
            self.target_system = HttpdServer(args_server)

        self.workload_controller = WorkloadController(args_server, args_workload, self.target_system)

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
            # Retrieve data from Cyber-Twin
            cached_result = self.retrieve_from_cyber_twin(config, factors)
            if cached_result:
                perf, evaluated_cost = cached_result
            else:
                ready = True
                try:
                    if not self.target_system.start_container():
                        print("Failed to start container")
                        ready = False

                    #  Pull config file (server.xml in system server) to local (default.xml in tuning server)
                    if not self.target_system.backup_config():
                        print("Failed to backup config")
                        ready = False
                    # Stop system container to avoid config file being lock;
                    self.target_system.stop_container()

                    # Push config file to make sure consistency
                    if not self.target_system.restore_config():
                        print("Failed to restore config.")
                        ready = False

                    # Set all knobs for the current config (local) and copy to docker
                    if not self.target_system.set_server_knobs(config):
                        print("Failed to set knobs.")
                        ready = False

                    # Start the container to reload the new configuration, ensuring the modification will work
                    if not self.target_system.start_container():
                        print("Failed to start server with new config.")
                        ready = False

                    # test whether the modification is successful
                    # self.target_system.backup_config()

                    # Retry connecting to Tomcat server
                    if not ServerConnector(self.password, self.server_url).connect_with_retry():
                        print("Failed to connect to server.")
                        ready = False

                except Exception as e:
                    print(f"Exception during preparation: {e}")
                    ready = False

                # Evaluate workload and record performance & cost (if not ready, return 0 0)

                # Remark: increasing the num_iter can improve the robustness/reliability  of measurement.
                num_iter = 1
                total_perf = 0
                total_evaluated_cost = 0
                for _ in range(num_iter):
                    start = time.time()
                    if ready:
                        RPS, TPR = self.workload_controller.run_workload(factors)
                    else:
                        RPS, TPR = 0, 0

                    end = time.time()
                    evaluated_cost = end - start
                    # accumulative perf result
                    if self.optimize_objective == 'RPS':
                        total_perf += RPS
                    elif self.optimize_objective == 'TPR':
                        total_perf += TPR

                    total_evaluated_cost += evaluated_cost

                perf = total_perf / num_iter

                evaluated_cost = total_evaluated_cost / num_iter

            evaluated_config_pop.append((config, perf, evaluated_cost))
            self.logger.logging_cyber_twin(config, perf, evaluated_cost, factors, self.cyber_twin_path, self.cyber_twin_file)

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

            for row in reader:
                stored_config = {header[i]: row[i] for i in range(num_knobs)}
                stored_fidelity = eval(row[fidelity_index])

                if stored_config == config and stored_fidelity == list(fidelity.values()):
                    perf = float(row[perf_index])
                    evaluated_cost = float(row[evaluated_cost_index])
                    return (perf, evaluated_cost)

        return None

