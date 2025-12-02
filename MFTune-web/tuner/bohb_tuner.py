import random
import time
from systems.httpd_server import HttpdServer
from systems.tomcat_server import TomcatServer
from workload import WorkloadController
from tqdm import tqdm
from utils.logger import Logger
from math import log, ceil
import numpy as np
import statsmodels.api as sm
import scipy.stats as sps
from utils.config_utils import ConfigUtils
from utils.server_connector import ServerConnector


class BOHBTuner:
    def __init__(self, args_server, args_workload, args_tune, run):
        super(BOHBTuner, self).__init__()
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
        self.log_file = 'BOHBTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        # Target system
        if self.sys_name == 'tomcat':
            self.target_system = TomcatServer(args_server)
        elif self.sys_name == 'httpd':
            self.target_system = HttpdServer(args_server)

        self.workload_controller = WorkloadController(args_server, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)

        self.hb_factor = 'duration'
        if self.args_workload['workload_bench'] == 'wrk':
            self.hb_factor = 'duration'
        elif self.args_workload['work_bench'] == 'ab':
            self.hb_factor = 'requests'

        # Parameters Settings for Algorithm
        self.default_factors = self.workload_controller.get_default_fidelity_factors()
        self.R_unit = 10  # Define resource unit
        self.R = self.default_factors[self.hb_factor] / self.R_unit  # Maximum resource unit per config (only time factor)
        self.eta = 2  # Define config down-sampling rate
        self.s_max = int(log(self.R) / log(self.eta))  # Maximum stage in HS, as well as maximum bracket
        self.B = (self.s_max + 1) * self.R  # budget per bracket, i.e., budget for each full HS process

        self.evaluated_configs = set()
        self.hf_evaluated_configs = []  # Store all evaluated results in full-fidelity stages
        self.consumed_cost = 0

        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")

        # KDE config
        self.random_fraction = 1/3  # random sampling rate
        self.num_samples = 64       # sampling for kde test
        self.top_n_percent = 15     # n_good percent. 
        self.min_points_in_model = len(self.target_system.knobs_info) + 1  # min points for fitting kde

    def tune_bohb(self):
        start_time = time.time()
        iteration = 1
        while self.consumed_cost < self.total_budget:
            print(f"Executing the {iteration}-th iteration of BOHB ...")
            # outer loop (Bracket)
            for s in reversed(range(self.s_max + 1)):
                # Initial config count and resource per config (stage 0)
                n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
                r = self.R * self.eta ** (-s)

                configs = self.sample_configs_bohb(n)  # Initialize n configs (random or TPE)

                # Inner loop (Successive Halving)
                for i in range(s + 1):
                    n_i = int(n * self.eta ** (-i))  # number of configs for current stage
                    if i == s:
                        r_i = int(self.R)  # Make sure the evaluation is full resource unit for the config in last stage
                    else:
                        r_i = int(r * self.eta ** i)  # resource unit per config for current stage

                    budget = r_i * self.R_unit
                    factors = self.default_factors.copy()
                    factors[self.hb_factor] = budget  # Control fidelity
                    print(f'{self.hb_factor}')
                    print(f"[Iteration: {iteration}] | [Bracket: {self.s_max - s + 1}] | [SH stage: {i + 1}] | [Configs: {n_i}] | [Budget: {budget}s]")
                    evaluated_configs, cost = self.evaluate_configs(configs, factors)

                    if self.consumed_cost >= self.total_budget:
                        print("resource exhaustion.")
                        break

                    if r_i == self.R:  # full resource used, add to high-fidelity record (for fitting model)
                        self.hf_evaluated_configs.extend(evaluated_configs)

                    if self.optimize_objective in ['throughput', 'RPS']:
                        evaluated_configs.sort(key=lambda x: x[1], reverse=True)
                    elif self.optimize_objective in ['latency', 'run_time']:
                        evaluated_configs.sort(key=lambda x: x[1])

                    configs = [config for config, _, _ in evaluated_configs[: max(1, n_i // self.eta)]]

            iteration += 1

        self.pbar.close()
        end_time = time.time()
        runtime = end_time - start_time
        print(f"[DONE] Total Cost: {self.consumed_cost:.2f} | Time: {runtime:.2f}s")
        self.logger.store_runtime_to_csv(runtime, self.log_path)


    def sample_configs_bohb(self, sample_size):
        configs = []
        for _ in range(sample_size):
            if len(self.hf_evaluated_configs) < self.min_points_in_model + 2 or np.random.rand() < self.random_fraction:
                configs.append(self.random_sample_one())
            else:
                configs.append(self.model_based_sample())
        return configs

    def random_sample_one(self):
        config = {}
        for key, val in self.target_system.knobs_info.items():
            if val['type'] == 'integer':
                config[key] = random.randint(val['min'], val['max'])
            elif val['type'] == 'float':
                config[key] = random.uniform(val['min'], val['max'])
            elif val['type'] == 'enum':
                config[key] = random.choice(val['enum_values'])
        return config

    def model_based_sample(self):
        """
        Sample a configuration using BOHB's KDE-based model.
        Builds two KDEs: one from top-performing configurations (good),
        and another from the rest (bad), and samples candidates to minimize g(x)/l(x) or maximize l(x)/g(x)
        g(x) -> bad ; l(x) -> good
        """

        # Extract configs and performance values
        configs = [c for c, perf, _ in self.hf_evaluated_configs]
        perfs = [perf for c, perf, _ in self.hf_evaluated_configs]

        # Convert config dicts to numerical vectors (enum encoded as index)
        vectors = np.array(self.preprocess_configs_with_knobs_info(configs, self.target_system.knobs_info))
        perfs = np.array(perfs)

        # Sort configurations by performance (depending on objective)
        if self.optimize_objective in ['latency', 'run_time']:
            sorted_idx = np.argsort(perfs)  # lower is better
        elif self.optimize_objective in ['throughput', 'RPS']:
            sorted_idx = np.argsort(-perfs)  # higher is better

        num_total = len(vectors)
        n_good = max(self.min_points_in_model, int(num_total * self.top_n_percent / 100))
        n_bad = max(self.min_points_in_model, num_total - n_good)

        train_data_good = vectors[sorted_idx[:n_good]]
        train_data_bad = vectors[sorted_idx[n_good:n_good + n_bad]]

        # Skip model-based sampling if sample size too small
        if train_data_good.shape[0] <= train_data_good.shape[1] or \
                train_data_bad.shape[0] <= train_data_bad.shape[1]:
            print("[Warning] Not enough samples to fit KDE, fallback to random.")
            return self.random_sample_one()

        print(f'[DEBUG]: n_good: {n_good} | n_bad: {n_bad} for KDE')
        # Build variable types string for KDE: 'c' for continuous, 'u' for enum
        vartypes = ''.join(['u' if v['type'] == 'enum' else 'c' for v in self.target_system.knobs_info.values()])

        # Fit KDEs on good and bad samples
        bw_estimation = 'normal_reference'
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=vartypes, bw=bw_estimation)
        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=vartypes, bw=bw_estimation)

        # Enforce minimum bandwidth for stability
        good_kde.bw = np.clip(good_kde.bw, 1e-3, None)
        bad_kde.bw = np.clip(bad_kde.bw, 1e-3, None)

        # Define acquisition function
        def acquisition(x):
            x = np.array(x).reshape(1, -1)
            return bad_kde.pdf(x)[0] / max(good_kde.pdf(x)[0], 1e-6)

        # Model-based sampling loop
        best_x = None
        best_acq = float('inf')
        for _ in range(self.num_samples):
            idx = np.random.randint(0, len(train_data_good))
            base = train_data_good[idx]
            candidate = np.random.normal(loc=base, scale=good_kde.bw)
            try:
                acq_score = acquisition(candidate)
                if np.isfinite(acq_score) and acq_score < best_acq:
                    best_acq = acq_score
                    best_x = candidate
            except Exception:
                continue

        # Fallback: if no valid sample found, use random
        if best_x is None:
            print("[Warning] model-based sample failed, fallback to random.")
            return self.random_sample_one()

        # Convert vector back to config dict (decode enum values)
        return self.vector_to_config(best_x)

    def vector_to_config(self, vector):
        """
        Convert a vector back into a config dictionary.
        Enum values are mapped back to strings.
        """
        config = {}
        for i, (key, info) in enumerate(self.target_system.knobs_info.items()):
            if info['type'] == 'enum':
                enum_values = info['enum_values']
                index = int(round(vector[i]))
                index = max(0, min(index, len(enum_values) - 1))  # clamp to valid range
                config[key] = enum_values[index]
            elif info['type'] == 'integer':
                config[key] = int(round(vector[i]))
            else:  # float
                config[key] = float(vector[i])
        return config

    def preprocess_configs_with_knobs_info(self, configs, knobs_info):
        """
        Convert list of config dicts into a 2D numerical array for KDE.
        Enum parameters are mapped to indices.
        """
        processed_configs = []
        for config in configs:
            processed_config = []
            for key in knobs_info:  # make sure the order is consistency
                value = config[key]
                if knobs_info[key]["type"] == "enum":
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


