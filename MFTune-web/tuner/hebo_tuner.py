import random
import numpy as np
import time
from systems.httpd_server import HttpdServer
from systems.tomcat_server import TomcatServer
from workload import WorkloadController
from tqdm import tqdm
from utils.logger import Logger
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace
from utils.server_connector import ServerConnector


class HEBOTuner:
    def __init__(self, args_server, args_workload, args_tune, run):
        super(HEBOTuner, self).__init__()
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
        self.log_file = 'HEBOTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'
        self.consumed_cost = 0
        self.evaluated_configs = set()

        # Target system
        if self.sys_name == 'tomcat':
            self.target_system = TomcatServer(args_server)
        elif self.sys_name == 'httpd':
            self.target_system = HttpdServer(args_server)

        self.workload_controller = WorkloadController(args_server, args_workload, self.target_system)
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
                    # RPS = random.uniform(300, 500)
                    # TPR = 1000.0 / RPS  # mock: inverse proportional
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
