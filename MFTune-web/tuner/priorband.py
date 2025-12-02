import time
import random
import numpy as np
from math import log, ceil

from pyDOE import lhs
from workload import WorkloadController
from systems.httpd_server import HttpdServer
from systems.tomcat_server import TomcatServer
from utils.server_connector import ServerConnector
from tqdm import tqdm
from utils.logger import Logger
from utils.config_utils import ConfigUtils
import math


class PriorBandTuner:
    def __init__(self, args_server, args_workload, args_tune, run):
        super(PriorBandTuner, self).__init__()
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
        self.log_file = 'PriorBand_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        # Target system
        if self.sys_name == 'tomcat':
            self.target_system = TomcatServer(args_server)
        elif self.sys_name == 'httpd':
            self.target_system = HttpdServer(args_server)

        self.workload_controller = WorkloadController(args_server, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)

        self.default_factors = self.workload_controller.get_default_fidelity_factors()
        self.R_unit = 10

        self.hb_factor = 'duration'
        if self.args_workload['workload_bench'] == 'wrk':
            self.hb_factor = 'duration'
        elif self.args_workload['work_bench'] == 'ab':
            self.hb_factor = 'requests'

        self.R = self.default_factors[self.hb_factor] / self.R_unit
        self.eta = 2
        self.s_max = int(log(self.R) / log(self.eta))
        self.B = (self.s_max + 1) * self.R

        self.evaluated_configs = set()
        self.consumed_cost = 0
        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")

        self.knobs_info = self.target_system.knobs_info
        self.prior_mean = self.target_system.get_default_knobs()
        self.incumbent_config = None
        self.incumbent_perf = None

        self.dynamic_adjust_enabled = False
        self.history_by_rung = dict()  # stage index i -> list of (config, perf, cost)
        self.p_pi_old = 0.0  # previous p_pi value for dynamic adjustment

    def tune_priorband(self):
        start_time = time.time()
        # Continuously run the HyperBand within the constraint of total budget
        iteration = 1
        while self.consumed_cost < self.total_budget:
            print(f"Executing the {iteration}-th iteration of PriorBand ...")
            # outer loop (Bracket)
            for s in reversed(range(self.s_max + 1)):
                # Initial number of configs
                n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
                # Initial number of resource unit per config
                r = self.R * self.eta ** (-s)

                # --- PriorBand sampling ---
                rung = self.s_max - s  # initial stage index in each bracket
                configs = self.sample_configs_priorband(n, rung)

                for i in range(s + 1):
                    n_i = int(n * self.eta ** (-i))
                    r_i = int(self.R if i == s else r * self.eta ** i)

                    budget = r_i * self.R_unit
                    factors = self.default_factors.copy()
                    factors[self.hb_factor] = budget
                    print(f"[Iteration: {iteration}] | [Bracket: {self.s_max - s + 1}] | [SH stage: {i + 1}] | [Configs: {n_i}] | [Budget: {budget}s]")

                    evaluated_configs, _ = self.evaluate_configs(configs, factors)

                    if self.consumed_cost >= self.total_budget:
                        print("Resource exhausted.")
                        break

                    # update incumbent if evaluated at max fidelity
                    for config, perf, cost in evaluated_configs:
                        if factors[self.hb_factor] == self.default_factors[self.hb_factor]:
                            if self.incumbent_config is None or self._is_better(perf, self.incumbent_perf):
                                self.incumbent_config = config
                                self.incumbent_perf = perf
                                self.dynamic_adjust_enabled = True

                    if i not in self.history_by_rung:
                        self.history_by_rung[i] = []
                    self.history_by_rung[i].extend(evaluated_configs)

                    evaluated_configs.sort(key=lambda x: x[1], reverse=(self.optimize_objective in ['throughput', 'RPS']))
                    configs = [config for config, _, _ in evaluated_configs[: max(1, n_i // self.eta)]]

        print(f'Total consumed time budget: {self.consumed_cost}')
        self.logger.store_runtime_to_csv(time.time(), self.log_path)
        self.pbar.close()

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

    def _is_better(self, p1, p2):
        if p2 is None:
            return True
        return p1 > p2 if self.optimize_objective in ['throughput', 'RPS'] else p1 < p2

    def sample_configs_priorband(self, n, rung):
        eta_pow = self.eta ** rung
        p_U = 1 / (1 + eta_pow)
        p_mix = 1 - p_U

        configs = []
        n_U = int(n * p_U)
        n_remaining = n - n_U

        if self.dynamic_adjust_enabled and self.incumbent_config is not None:
            scores = self._compute_weight_scores(rung)
            total = scores['prior'] + scores['incumbent']
            # p_pi = self.p_pi_old * scores['prior'] / total
            # p_lh = self.p_pi_old * scores['incumbent'] / total
            p_pi = p_mix * scores['prior'] / total
            p_lh = p_mix * scores['incumbent'] / total
        else:
            p_pi = p_mix
            p_lh = 0

        # self.p_pi_old = p_pi

        n_pi = int(n_remaining * p_pi / (p_pi + p_lh)) if (p_pi + p_lh) > 0 else 0
        n_lh = n_remaining - n_pi

        # 修正误差
        total_samples = n_U + n_pi + n_lh
        if total_samples < n:
            deficit = n - total_samples
            n_lh += deficit

        if n_U > 0:
            configs += ConfigUtils.sampling_configs_by_rs(n_U, self.knobs_info)
        if n_pi > 0:
            configs += self._sample_from_prior(n_pi)
        if n_lh > 0 and self.incumbent_config:
            configs += self._sample_from_incumbent(n_lh)

        return configs[:n]

    def _compute_weight_scores(self, rung):
        """
        Compute weighted density scores for prior and incumbent for a given rung,
        based on PriorBand Algorithm 2.
        """
        # fallback: not enough history
        if rung not in self.history_by_rung:
            return {'prior': 1.0, 'incumbent': 0.0}

        evaluated = self.history_by_rung[rung]
        if len(evaluated) < self.eta:
            return self._compute_weight_scores(rung - 1) if rung > 0 else {'prior': 1.0, 'incumbent': 0}

        # Select top-1/eta configurations
        top_k = max(len(evaluated) // self.eta, self.eta)
        reverse = self.optimize_objective in ['throughput', 'RPS']
        top_configs = sorted(evaluated, key=lambda x: x[1], reverse=reverse)[:top_k]

        # Weighted scoring: w_i = top_k + 1 - i
        weights = [top_k + 1 - i for i in range(1, top_k + 1)]
        S_pi = 0.0
        S_lh = 0.0

        for i, (config, _, _) in enumerate(top_configs):
            w = weights[i]
            try:
                prior_score = self._prior_density(config)
            except:
                prior_score = 1e-6
            try:
                inc_score = self._incumbent_density(config)
            except:
                inc_score = 1e-6
            S_pi += w * prior_score
            S_lh += w * inc_score

        return {'prior': S_pi, 'incumbent': S_lh}

    def _prior_density(self, config):
        return self._config_density(config, self.prior_mean)

    def _incumbent_density(self, config):
        return self._config_density(config, self.incumbent_config)

    def _config_density(self, config, center_config):
        density = 1.0
        sigma = 0.25
        for name, info in self.knobs_info.items():
            val = config[name]
            center = center_config[name]
            try:
                if info['type'] == 'float':
                    val = float(val)
                    center = float(center)
                    density *= self._gaussian_pdf(val, center, sigma)
                elif info['type'] == 'integer':
                    val = int(val)
                    center = int(center)
                    density *= self._gaussian_pdf(val, center, sigma)
                elif info['type'] == 'enum':
                    density *= 1.0 if str(val) == str(center) else 1.0 / (len(info["enum_values"]) - 1 + 1e-6)
            except:
                density *= 1e-6
        return density

    def _gaussian_pdf(self, x, mu, sigma):
        coef = 1.0 / (sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return coef * np.exp(exponent)


    def _sample_from_prior(self, n):
        sampled = []
        for _ in range(n):
            config = {}
            for name, info in self.knobs_info.items():
                default = self.prior_mean[name]
                knob_type = info["type"]

                if knob_type == "float":
                    low, high = float(info["min"]), float(info["max"])
                    val = np.random.normal(loc=default, scale=0.25)
                    if low <= val <= high:
                        config[name] = float(val)
                    else:
                        config[name] = float(np.random.uniform(low, high))

                elif knob_type == "integer":
                    low, high = int(info["min"]), int(info["max"])
                    val = int(np.round(np.random.normal(loc=default, scale=0.25)))
                    if low <= val <= high:
                        config[name] = val
                    else:
                        config[name] = random.randint(low, high)

                elif knob_type == "enum":
                    enum_vals = info["enum_values"]
                    if random.random() < 0.25:
                        config[name] = random.choice([v for v in enum_vals if v != str(default)])
                    else:
                        config[name] = str(default)
            sampled.append(config)
        return sampled

    def _sample_from_incumbent(self, n):
        sampled = []
        for _ in range(n):
            config = {}
            for name, info in self.knobs_info.items():
                base = self.incumbent_config[name]
                knob_type = info['type']

                # 50% 概率进行扰动
                if random.random() < 0.5:
                    if knob_type == 'float':
                        low, high = float(info['min']), float(info['max'])
                        val = np.random.normal(loc=float(base), scale=0.25)
                        if low <= val <= high:
                            config[name] = float(val)
                        else:
                            config[name] = float(np.random.uniform(low, high))

                    elif knob_type == 'integer':
                        low, high = int(info['min']), int(info['max'])
                        val = int(np.round(np.random.normal(loc=int(base), scale=0.25)))
                        if low <= val <= high:
                            config[name] = val
                        else:
                            config[name] = random.randint(low, high)


                    elif knob_type == 'enum':
                        enum_vals = list(info['enum_values'])
                        k = len(enum_vals)
                        # 按论文：incumbent 权重 k，其余各 1
                        weights = []
                        for v in enum_vals:
                            weights.append(k if str(v) == str(base) else 1)
                        # 归一化为概率
                        s = float(sum(weights))
                        probs = [w / s for w in weights]
                        # 采样（可能留在 incumbent，符合“局部扰动”）
                        config[name] = random.choices(enum_vals, weights=probs, k=1)[0]
                else:
                    # 保持原值
                    config[name] = base

            sampled.append(config)
        return sampled
