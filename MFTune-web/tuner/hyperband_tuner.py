import time
from pyDOE import lhs

from systems.httpd_server import HttpdServer
from systems.tomcat_server import TomcatServer
from utils.server_connector import ServerConnector
from workload import WorkloadController
from tqdm import tqdm
from utils.logger import Logger
from math import log, ceil


class HBTuner:
    def __init__(self, args_server, args_workload, args_tune, run):
        super(HBTuner, self).__init__()
        self.max_iter = int(args_tune['max_iter'])
        self.total_budget = int(args_tune['total_budget'])

        self.args_server = args_server
        self.password = args_server['password']
        self.server_url = args_server['url']

        self.args_workload = args_workload
        self.args_tune = args_tune
        self.workload_bench = args_workload["workload_bench"]
        self.tuning_method = args_tune['tuning_method']
        self.fidelity_type = args_tune['fidelity_type']
        self.fidelity_metric = args_tune['fidelity_metric']
        self.optimize_objective = args_tune['optimize_objective']
        self.sys_name = self.args_server['server']
        self.log_path = f'experimental_results/{self.sys_name}/{self.workload_bench}/{self.tuning_method}/run_{run}_{self.tuning_method}_{self.fidelity_type}'
        self.log_file = 'HBTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        # Target system
        if self.sys_name == 'tomcat':
            self.target_system = TomcatServer(args_server)
        elif self.sys_name == 'httpd':
            self.target_system = HttpdServer(args_server)

        # Workload Controller; Logger;
        self.workload_controller = WorkloadController(args_server, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)

        # Parameters Settings for Algorithm
        self.default_factors = self.workload_controller.get_default_fidelity_factors()
        self.R_unit = 10  # Define resource unit

        self.hb_factor = 'duration'
        if self.args_workload['workload_bench'] == 'wrk':
            self.hb_factor = 'duration'
        elif self.args_workload['work_bench'] == 'ab':
            self.hb_factor = 'requests'

        self.R = self.default_factors[self.hb_factor] / self.R_unit  # Maximum resource unit per config (only time factor)
        self.eta = 2  # Define config down-sampling rate
        self.s_max = int(log(self.R) / log(self.eta))  # Maximum stage in HS, as well as maximum bracket
        self.B = (self.s_max + 1) * self.R  # budget per bracket, i.e., budget for each full HS process

        self.evaluated_configs = set()
        self.consumed_cost = 0

    def tune_hyperband(self):

        start_time = time.time()
        print("Start running hyperband...")
        pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")
        print("...")

        # Continuously run the HyperBand within the constraint of total budget
        while self.consumed_cost < self.total_budget:
            print("Executing a new round of HyperBand...")

            # outer loop (HyperBand)
            for s in reversed(range(self.s_max + 1)):
                # Initial number of configs
                n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
                # Initial number of resource unit per config
                r = self.R * self.eta ** (-s)

                # Initialize n configs uniformly
                configs = self.sampling_configs_by_lhs(n)
                factors = self.default_factors

                # Inner loop (Successive Halving)
                for i in range(s + 1):
                    n_i = int(n * self.eta ** (-i))  # number of configs for current stage
                    if i == s:
                        r_i = int(self.R)  # Make sure the evaluation is full resource unit for the config in last stage
                    else:
                        r_i = int(r * self.eta ** i)  # resource unit per config for current stage

                    print(f"Bracket {s}; SH stage {i}; Configs {n_i}; resources each {r_i}.")

                    factors[self.hb_factor] = r_i * self.R_unit
                    evaluated_configs, cost = self.evaluate_configs(configs, factors)
                    pbar.update(cost)

                    # Resource exhaustion
                    if self.consumed_cost >= self.total_budget:
                        print("resource exhaustion.")
                        break

                    # Select a number of (n/eta) the best configus for the next HS loop (next stage)
                    if self.optimize_objective == 'RPS':
                        evaluated_configs.sort(key=lambda x: x[1], reverse=True)
                    elif self.optimize_objective == 'TPR':
                        evaluated_configs.sort(key=lambda x: x[1])

                    configs = [config for config, _, _ in evaluated_configs[: max(1, n_i // self.eta)]]

        print(f'total consume time budget: {self.consumed_cost}')

        end_time = time.time()
        runtime = end_time - start_time
        pbar.close()


        # Record runtime using the Logger class
        self.logger.store_runtime_to_csv(runtime, self.log_path)

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

    def evaluate_configs(self, configs, factors):
        """
        :param configs: config(s) that need to be evaluated (list)
        :param factors: represents a specific fidelity setting
        :return:
        """
        evaluated_configs = []
        cost_configs = 0
        for config in configs:
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

            num_iter = 1
            total_perf = 0
            total_cost = 0
            for _ in range(num_iter):
                start = time.time()
                if ready:
                    RPS, TPR = self.workload_controller.run_workload(factors)
                else:
                    RPS, TPR = 0, 0
                end = time.time()
                duration = end - start

                if self.optimize_objective == 'RPS':
                    total_perf += RPS
                elif self.optimize_objective == 'TPR':
                    total_perf += TPR

                total_cost += duration

            avg_perf = total_perf / num_iter
            avg_cost = total_cost / num_iter

            self.consumed_cost += avg_cost
            cost_configs += avg_cost

            evaluated_configs.append((config, avg_perf, avg_cost))

            self.logger.logging_data(config, avg_perf, avg_cost, factors, self.log_path, self.log_file)
            self.logger.logging_cyber_twin(config, avg_perf, avg_cost, factors, self.cyber_twin_path,
                                           self.cyber_twin_file)

            if self.consumed_cost >= self.total_budget:
                print("Total budget exhausted.")
                break

        return evaluated_configs, cost_configs

