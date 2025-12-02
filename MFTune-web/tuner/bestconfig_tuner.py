import math
from tqdm import tqdm
import random
import time

from systems.httpd_server import HttpdServer

from systems.tomcat_server import TomcatServer
from utils.server_connector import ServerConnector
from workload import WorkloadController
from utils.logger import Logger


class BestConfigTuner:
    def __init__(self, args_server, args_workload, args_tune, run):
        super(BestConfigTuner, self).__init__()
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
        self.log_file = 'BestConfigTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        # Parameters Settings for Algorithm: Flash for config space
        self.max_rounds = 5
        self.sampling_size = 8
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

    def tune_bestconfig(self):
        start_time = time.time()

        if self.fidelity_type == 'single_fidelity':
            hf_factors = self.workload_controller.get_default_fidelity_factors()
            self.bestconfig_search_config(self.total_budget, hf_factors)
        elif self.fidelity_type == 'multi_fidelity':
            pass

        end_time = time.time()
        runtime = end_time - start_time

        # Record runtime using the Logger class
        self.logger.store_runtime_to_csv(runtime, self.log_path)


    def bestconfig_search_config(self, budget, fidelity, init_configs=None, kd_corr=1, fidelity_id=0, evaluated_filtered_configs=None):
        """
        :param budget:
        :param fidelity:
        :param init_configs:
        :param kd_corr:
        :param fidelity_id:
        :param lf_filtered_configs:
        :return:
        """

        print("Start running BestConfig...")
        pbar = tqdm(total=budget, desc="Tuning Progress", unit="cost")
        print("...")

        consumed_cost = 0
        best_config = None
        best_perf = -math.inf if self.optimize_objective == 'RPS' else math.inf

        # Initial DDS sampling
        sampled_configs = self.dds_sampling(self.sampling_size)
        evaluated_configs, cost_init_configs = self.evaluate_configs(sampled_configs, fidelity)
        consumed_cost += cost_init_configs

        pbar.update(cost_init_configs)

        # if there exist configs that have been evaluated by low fidelity, evaluated it under current fidelity and
        # combine it with initial configs
        if evaluated_filtered_configs:

            # Combine and sort the results from both evaluations
            evaluated_configs = evaluated_configs + evaluated_filtered_configs
            if self.optimize_objective == 'RPS':
                evaluated_configs.sort(key=lambda x: x[1], reverse=True)
            elif self.optimize_objective == 'TPR':
                evaluated_configs.sort(key=lambda x: x[1])

        evaluated_configs = evaluated_configs[:self.sampling_size]

        # initialize the best config
        for config, perf, _ in evaluated_configs:
            if (self.optimize_objective == 'RPS' and perf > best_perf) or \
                    (self.optimize_objective == 'TPR' and perf < best_perf):
                best_perf = perf
                best_config = config

        while consumed_cost < budget:
            found_better = False
            for round_counter in range(self.max_rounds):
                # RBS is used to determine the boundary
                bounded_space = self.define_bounded_space(best_config, evaluated_configs)

                # sampling configs under the boundary
                bounded_sampled_configs = self.dds_sampling_within_bounds(bounded_space, self.sampling_size)
                new_evaluated_configs, cost_new_configs = self.evaluate_configs(bounded_sampled_configs, fidelity)
                consumed_cost += cost_new_configs
                pbar.update(cost_new_configs)

                # update the best config
                for config, perf, _ in new_evaluated_configs:
                    if (self.optimize_objective == 'RPS' and perf > best_perf) or \
                            (self.optimize_objective == 'TPR' and perf < best_perf):
                        best_perf = perf
                        best_config = config
                        found_better = True

                if consumed_cost >= budget:
                    return best_config, best_perf

                if not found_better:
                    break

            # if budget are not exhausted, resampling some configs from scratch
            sampled_configs = self.dds_sampling(self.sampling_size)
            evaluated_configs, cost_global_configs = self.evaluate_configs(sampled_configs, fidelity)
            consumed_cost += cost_global_configs
            pbar.update(cost_global_configs)

            for config, perf, _ in evaluated_configs:
                if (self.optimize_objective == 'RPS' and perf > best_perf) or \
                        (self.optimize_objective == 'TPR' and perf < best_perf):
                    best_perf = perf
                    best_config = config
        pbar.close()

        return best_config, best_perf

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

    def sampling_configs(self, sampling_size):
        """sampling configs randomly,make ensure without repeat configs"""
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
                # TODO: extend other types of configs
            config_tuple = tuple(config.items())
            if config_tuple not in seen_configs:
                init_configs.append(config)
                seen_configs.add(config_tuple)

        return init_configs

    def dds_sampling(self, sampling_size):
        """
        Use Divide & Diverge Sampling (DDS) to generate configs
        each dimension is divided into sampling_size area and chose a value within each area
        """
        # obtain the partitions of each parameter
        knobs_info = self.target_system.knobs_info
        partitions = {knob_name: self.get_intervals(knob_info, sampling_size)
                      for knob_name, knob_info in knobs_info.items()}

        # initialize selectable indices of each dimension
        available_indices = {knob_name: list(range(sampling_size)) for knob_name in partitions.keys()}

        samples = []
        for _ in range(sampling_size):
            config = {}
            for knob_name, intervals in partitions.items():

                index = random.choice(available_indices[knob_name])
                lower, upper = intervals[index]
                if knobs_info[knob_name]['type'] == 'integer':
                    value = round(random.uniform(lower, upper))
                elif knobs_info[knob_name]['type'] == 'float':
                    value = random.uniform(lower, upper)
                elif knobs_info[knob_name]['type'] == 'enum':
                    value = self.map_to_enum(index, knobs_info[knob_name]['enum_values'], sampling_size)
                else:
                    raise ValueError(f"Unsupported knob type: {knobs_info[knob_name]['type']}")
                config[knob_name] = value

                available_indices[knob_name].remove(index)
            samples.append(config)

        return samples

    @staticmethod
    def get_intervals(knob_info, sampling_size):
        """
        divide parameter range into different area according to knob's type
        """
        if knob_info['type'] in ['integer', 'float']:
            step = (knob_info['max'] - knob_info['min']) / sampling_size
            return [(knob_info['min'] + step * i, knob_info['min'] + step * (i + 1)) for i in range(sampling_size)]
        elif knob_info['type'] == 'enum':

            enum_values = knob_info['enum_values']
            num_values = len(enum_values)
            step = sampling_size // num_values
            intervals = []
            for i in range(sampling_size):
                value_index = min(i // step, num_values - 1)
                intervals.append((value_index, value_index))
            return intervals
        else:
            raise ValueError(f"Unsupported knob type: {knob_info['type']}")


    @staticmethod
    def map_to_enum(index, enum_values, sampling_size):
        """
        mapping sample area into concrete enum value
        """
        num_values = len(enum_values)
        step = sampling_size // num_values
        enum_index = min(index // step, num_values - 1)
        return enum_values[enum_index]


    def dds_sampling_within_bounds(self, bounded_space, sampling_size):
        """
        use DDS to sample configs within boundary
        """
        knobs_info = self.target_system.knobs_info
        partitions = {knob_name: self.get_partitions_within_bounds(knob_name, bounds, sampling_size)
                      for knob_name, bounds in bounded_space.items()}

        # initialize selectable indices of each dimension
        available_indices = {knob_name: list(range(sampling_size)) for knob_name in partitions.keys()}
        samples = []

        for _ in range(sampling_size):
            config = {}
            for knob_name, intervals in partitions.items():
                index = random.choice(available_indices[knob_name])
                lower, upper = intervals[index]

                if knobs_info[knob_name]['type'] == 'integer':
                    value = round(random.uniform(lower, upper))
                elif knobs_info[knob_name]['type'] == 'float':
                    value = random.uniform(lower, upper)
                elif knobs_info[knob_name]['type'] == 'enum':
                    value = self.map_to_enum(index, knobs_info[knob_name]['enum_values'], sampling_size)
                else:
                    raise ValueError(f"Unsupported knob type: {knobs_info[knob_name]['type']}")
                config[knob_name] = value
                available_indices[knob_name].remove(index)
            samples.append(config)

        return samples

    def get_partitions_within_bounds(self, knob_name, bounds, sampling_size):
        """
        :param bounds: (lower_bound, upper_bound)
        :param knob_name
        :param knob_info:
        :param sampling_size:
        :return:
        """
        knob_info = self.target_system.knobs_info[knob_name]
        lower_bound, upper_bound = bounds

        if knob_info['type'] in ['integer', 'float']:
            step = (upper_bound - lower_bound) / sampling_size
            return [(lower_bound + step * i, lower_bound + step * (i + 1)) for i in range(sampling_size)]

        elif knob_info['type'] == 'enum':

            enum_values = knob_info['enum_values']
            num_values = len(enum_values)

            step = sampling_size // num_values
            intervals = []
            for i in range(sampling_size):
                value_index = min(i // step, num_values - 1)
                intervals.append((value_index, value_index))
            return intervals
        else:
            raise ValueError(f"Unsupported knob type: {knob_info['type']}")


    def define_bounded_space(self, best_config, evaluated_configs):
        """
        Define RBS boundary according to current best config and historical measurements.
        :param best_config: current best config
        :param evaluated_configs: historical measurements [(config, perf, cost), ...]
        :return: boundary of each dimension {knob_name: (lower_bound, upper_bound)}
        """
        bounded_space = {}

        for knob_name, best_value in best_config.items():
            knob_info = self.target_system.knobs_info.get(knob_name, {})
            knob_type = knob_info.get("type", "float")

            if knob_type == "enum":
                enum_values = knob_info.get("enum_values", [])
                if best_value not in enum_values:
                    # fallback for safety
                    bounded_space[knob_name] = (best_value, best_value)
                    continue

                idx = enum_values.index(best_value)
                lower_bound = enum_values[idx - 1] if idx > 0 else best_value
                upper_bound = enum_values[idx + 1] if idx < len(enum_values) - 1 else best_value

            else:
                lower_bound = max([config[knob_name] for config, _, _ in evaluated_configs if config[knob_name] < best_value], default=best_value)
                upper_bound = min([config[knob_name] for config, _, _ in evaluated_configs if config[knob_name] > best_value], default=best_value)

            bounded_space[knob_name] = (lower_bound, upper_bound)

        return bounded_space
