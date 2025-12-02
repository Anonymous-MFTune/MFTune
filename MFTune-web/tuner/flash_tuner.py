import random
import time
from systems.httpd_server import HttpdServer
from systems.tomcat_server import TomcatServer
from workload import WorkloadController
from tqdm import tqdm
from utils.logger import Logger
from sklearn.tree import DecisionTreeRegressor
from utils.server_connector import ServerConnector
from utils.config_utils import ConfigUtils

class FLASHTuner:
    def __init__(self, args_server, args_workload, args_tune, run):
        super(FLASHTuner, self).__init__()
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
        self.log_file = 'FLASHTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        # Parameters Settings for Algorithm: Flash for config space
        self.initial_size = 30
        self.sampling_size = 1000
        self.evaluated_configs = set()
        self.consumed_cost = 0

        # Target system
        if self.sys_name == 'tomcat':
            self.target_system = TomcatServer(args_server)
        elif self.sys_name == 'httpd':
            self.target_system = HttpdServer(args_server)

        # Workload Controller; Logger;
        self.workload_controller = WorkloadController(args_server, args_workload, self.target_system)
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
            init_configs = ConfigUtils.sampling_configs_by_rs(self.initial_size, self.target_system.knobs_info)
        evaluated_configs, cost_init_configs = self.evaluate_configs(init_configs, fidelity)
        consumed_cost += cost_init_configs
        pbar.update(cost_init_configs)

        # prepared for extension to multi-fidelity
        if evaluated_filtered_configs:
            # Combine and sort the results from both evaluations
            evaluated_configs = evaluated_configs + evaluated_filtered_configs
            if self.optimize_objective == 'RPS':
                evaluated_configs.sort(key=lambda x: x[1], reverse=True)
            elif self.optimize_objective == 'TPR':
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

            if self.optimize_objective == 'RPS':
                best_config_index = predicted_performances.argmax()
            elif self.optimize_objective == 'TPR':
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

