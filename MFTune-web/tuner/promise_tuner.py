import time
import random

import numpy as np
import pandas as pd
from itertools import product
import copy
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import lingam
from causallearn.search.ConstraintBased.FCI import fci
from utils.server_connector import ServerConnector
from systems.httpd_server import HttpdServer
from systems.tomcat_server import TomcatServer
from workload import WorkloadController
from utils.logger import Logger
from tqdm import tqdm
from utils.config_utils import ConfigUtils


class PromiseTuner:
    def __init__(self, args_server, args_workload, args_tune, run):
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
        self.log_file = 'PromiseTuner_results.csv'
        self.cyber_twin_path = f'experimental_results/{self.sys_name}/{self.workload_bench}'
        self.cyber_twin_file = 'cyber-twin.csv'

        self.initial_size = 10
        self.l_value = 10
        self.k_percent = 0.1
        self.consumed_cost = 0
        self.evaluated_data = []  # [(config, perf, cost)]
        self.evaluated_configs = set()  # Set of (config tuple, fidelity tuple)

        # Target system
        if self.sys_name == 'tomcat':
            self.target_system = TomcatServer(args_server)
        elif self.sys_name == 'httpd':
            self.target_system = HttpdServer(args_server)

        self.workload_controller = WorkloadController(args_server, args_workload, self.target_system)
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)
        self.pbar = tqdm(total=self.total_budget, desc="Tuning Progress", unit="cost")

    def tune_promise(self):
        start_time = time.time()
        hf_factors = self.workload_controller.get_default_fidelity_factors()
        self.promise_search_config(self.total_budget, hf_factors)
        end_time = time.time()
        self.logger.store_runtime_to_csv(end_time - start_time, self.log_path)
        self.pbar.close()

    def promise_search_config(self, budget, fidelity):

        init_configs = ConfigUtils.sampling_configs_by_rs(self.initial_size, self.target_system.knobs_info)
        self.evaluate_and_record(init_configs, fidelity)

        while self.consumed_cost < budget:
            rules = self.extract_rules_from_rf()
            rule_features, labels = self.featurize_rules(rules)
            purified_rules = self.purify_rules_by_causality(rule_features, labels, rules)
            candidate_pool = self.sample_candidates_with_rules(purified_rules, fidelity)
            best_config = self.select_by_ei(candidate_pool, fidelity, self.target_system.knobs_info)
            self.evaluate_and_record([best_config], fidelity)

    def sample_random_configs(self, sampling_size, fidelity, knobs_info):
        random.seed(time.time())
        init_configs = []
        seen_configs = set()
        while len(init_configs) < sampling_size:
            config = {}
            for knob_name, knob_info in knobs_info.items():
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

    def evaluate_and_record(self, configs, fidelity, stage_budget=None):
        evaluated_configs, cost = self.evaluate_configs(configs, fidelity)
        for config, perf, evaluated_cost in evaluated_configs:
            key = (tuple(sorted(config.items())), tuple(sorted(fidelity.items())))
            self.evaluated_configs.add(key)
            self.evaluated_data.append((config, perf, evaluated_cost))
        return evaluated_configs, cost

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


    def extract_rules_from_rf(self):
        configs = [c for c, _, _ in self.evaluated_data]
        y = [p for _, p, _ in self.evaluated_data]
        feature_names = list(self.target_system.knobs_info.keys())
        rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=self.l_value)
        x = self.preprocess_configs(configs)
        rf.fit(x, y)
        all_rules = []
        for tree in rf.estimators_:
            rules = self.get_tree_paths(tree, feature_names)
            all_rules.extend([r for r in rules if r not in all_rules])
        return all_rules

    def get_tree_paths(self, tree, feature_names):
        paths = []
        stack = [(0, [])]
        while stack:
            node_id, path = stack.pop()
            feature_index = tree.tree_.feature[node_id]
            threshold = tree.tree_.threshold[node_id]
            if feature_index != -2:
                name = feature_names[feature_index]
                stack.append((tree.tree_.children_left[node_id], path + [(name, threshold, 'L')]))
                stack.append((tree.tree_.children_right[node_id], path + [(name, threshold, 'R')]))
            else:
                paths.append(path)
        return paths

    def preprocess_configs(self, configs):
        knobs_info = self.target_system.knobs_info
        processed = []
        for config in configs:
            row = []
            for k in knobs_info:
                val = config[k]
                if knobs_info[k]['type'] == 'enum':
                    row.append(knobs_info[k]['enum_values'].index(val))
                else:
                    row.append(val)
            processed.append(row)
        return processed

    def featurize_rules(self, rules):
        configs = [c for c, _, _ in self.evaluated_data]
        feature_names = list(self.target_system.knobs_info.keys())
        feature_matrix = []
        for config in configs:
            row = []
            for rule in rules:
                row.append(1 if self.config_in_rule(config, rule, feature_names) else 0)
            feature_matrix.append(row)
        labels = [p for _, p, _ in self.evaluated_data]
        return feature_matrix, labels

    def config_in_rule(self, config, rule, feature_names):
        knobs_info = self.target_system.knobs_info
        for key, threshold, direction in rule:
            val = config[key]
            if knobs_info[key]['type'] == 'enum':
                val = knobs_info[key]['enum_values'].index(val)  # 枚举值转成 index

            if direction == 'L' and not val <= threshold:
                return False
            if direction == 'R' and not val > threshold:
                return False
        return True

    def purify_rules_by_causality(self, X, y, rules):
        """
        Perform causal filtering using DirectLiNGAM on rule-feature matrix.
        Rules with negative ACE (average causal effect) on performance are retained.
        """
        # 1. 构造 DataFrame：每列是一个 rule_feature，最后一列是 performance
        df = pd.DataFrame(X, columns=[f'rule_{i}' for i in range(len(rules))])
        df['perf'] = y
        data = df.to_numpy()
        num_features = data.shape[1]

        # 2. 初始化因果结构 prior_knowledge
        try:
            graph, _ = fci(data, show_progress=False, verbose=False)
            adjacency_matrix = self.get_adjacency_matrix(graph)
            if adjacency_matrix.shape != (num_features, num_features):
                raise ValueError("Adjacency matrix shape mismatch.")
            adjacency_matrix = adjacency_matrix.T
            adjacency_matrix[:, -1] = 0  # 保证 performance 没有 outgoing edge
        except Exception as e:
            print(f"[DEBUG] FCI failed or matrix invalid: {e}")
            adjacency_matrix = np.zeros((num_features, num_features), dtype=int)

        # 3. 构造 prior_knowledge
        prior_knowledge = adjacency_matrix.copy()
        prior_knowledge[-1, :-1] = 1  # perf 可以被所有 rule 影响
        prior_knowledge[-1, -1] = -1  # perf 不影响任何其他变量

        # 4. 利用 DirectLiNGAM 拟合因果模型
        purified = []
        try:
            model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
            model.fit(data)
            target_idx = num_features - 1
            print(f"[DEBUG] Num samples: {len(y)}, Num rules: {len(rules)}")
            for i in range(len(rules)):
                ace = model.estimate_total_effect(data, i, target_idx)
                print(f"[DEBUG] ACE[{i}]: {ace:.4f}")
                if ace < 0:
                    purified.append(rules[i])
        except Exception as e:
            print(f"[DEBUG] Causality analysis failed: {e}")

        return purified

    def get_adjacency_matrix(self, graph):
        nodes = graph.get_nodes()
        node_count = len(nodes)
        matrix = np.zeros((node_count, node_count), dtype=int)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        for edge in graph.get_graph_edges():
            i = node_index[edge.node1]
            j = node_index[edge.node2]
            matrix[i, j] = 1
        return matrix

    def sample_candidates_with_rules(self, rules, fidelity):
        knobs_info = self.target_system.knobs_info
        feature_names = list(knobs_info.keys())
        all_candidates = []

        configs = [c for c, _, _ in self.evaluated_data]
        perfs = [p for _, p, _ in self.evaluated_data]
        X = self.preprocess_configs(configs)
        model = RandomForestRegressor()
        model.fit(X, perfs)
        estimators = model.estimators_
        eta = min(perfs) if self.optimize_objective in ['latency', 'run_time'] else max(perfs)

        # fallback if no rule
        if not rules:
            candidates, _ = self.ei_guided_sampling_from_domain(knobs_info, estimators, eta, fidelity)
            return candidates

        for rule in rules:
            modified_knobs_info = self.modify_knob_info_by_rule(rule, knobs_info)
            if modified_knobs_info is None:
                continue
            candidates, _ = self.ei_guided_sampling_from_domain(modified_knobs_info, estimators, eta, fidelity)
            all_candidates.extend(candidates)

        return all_candidates

    def modify_knob_info_by_rule(self, rule, original_knobs_info):
        knobs_info = {k: copy.deepcopy(v) for k, v in original_knobs_info.items()}
        if len(rule) > 1:
            print(f"[DEBUG] Applying rule: {rule} to knobs_info")
        for r_knob, threshold, direction in rule:
            if r_knob not in knobs_info:
                continue
            info = knobs_info[r_knob]

            # 修复 threshold 是 numpy 类型
            if isinstance(threshold, (np.ndarray, list)):
                threshold = threshold[0]

            if info['type'] == 'integer':
                threshold = int(threshold)
                if direction == 'L':
                    info['max'] = min(info['max'], threshold)
                elif direction == 'R':
                    info['min'] = max(info['min'], threshold + 1)

            elif info['type'] == 'float':
                threshold = float(threshold)
                if direction == 'L':
                    info['max'] = min(info['max'], threshold)
                elif direction == 'R':
                    info['min'] = max(info['min'], threshold)

            elif info['type'] == 'enum':
                enum_values = info['enum_values']
                try:
                    threshold_index = int(threshold)
                    if threshold_index < 0 or threshold_index >= len(enum_values):
                        continue
                    threshold_value = enum_values[threshold_index]
                except:
                    continue

                if direction == 'L':
                    info['enum_values'] = [v for v in enum_values if enum_values.index(v) <= threshold_index]
                elif direction == 'R':
                    info['enum_values'] = [v for v in enum_values if enum_values.index(v) > threshold_index]

                if not info['enum_values']:
                    return None

        return knobs_info

    def ei_guided_sampling_from_domain(self, knobs_info, estimators, eta, fidelity, max_iterations=10000, stop_threshold=0.01):
        feature_names = list(knobs_info.keys())
        values, configs, best = [], [], float('-inf')

        for i in range(max_iterations):
            # sampled_config = self.sample_random_configs(1, fidelity, knobs_info)[0]
            sampled_config = ConfigUtils.sampling_configs_by_rs(1, knobs_info)[0]  # Randomly sample a config
            x = self.preprocess_configs([sampled_config])[0]
            preds = [e.predict([x])[0] for e in estimators]
            mean = np.mean(preds)
            std = np.std(preds)
            if self.optimize_objective in ['latency', 'run_time']:  # minimization
                z = (eta - mean) / std if std > 0 else 0
                ei = (eta - mean) * norm.cdf(z) + std * norm.pdf(z) if std > 0 else 0
            else:  # throughput: maximization
                z = (mean - eta) / std if std > 0 else 0
                ei = (mean - eta) * norm.cdf(z) + std * norm.pdf(z) if std > 0 else 0

            values.append(ei)
            configs.append(sampled_config)
            best = max(best, ei)

            if i > 0:  # kde requires at least two points
                try:
                    kde = stats.gaussian_kde(values)
                    if 1 - kde.integrate_box_1d(-np.inf, best) < stop_threshold:
                        break
                except Exception as e:
                    break

        return configs, best

    def select_by_ei(self, candidates, fidelity, knobs_info):
        if not candidates:
            # return random.choice(self.sample_random_configs(1, fidelity, knobs_info))
            return random.choice(ConfigUtils.sampling_configs_by_rs(1, knobs_info))
        train_x = self.preprocess_configs([c for c, _, _ in self.evaluated_data])
        train_y = [p for _, p, _ in self.evaluated_data]
        model = RandomForestRegressor()
        model.fit(train_x, train_y)

        test_x = self.preprocess_configs(candidates)
        preds = np.array([e.predict(test_x) for e in model.estimators_]).T
        mean = preds.mean(axis=1)
        std = preds.std(axis=1)
        std[std == 0] = 1e-8
        best_seen = min(train_y) if self.optimize_objective in ['latency', 'run_time'] else max(train_y)
        if self.optimize_objective in ['latency', 'run_time']:  # minimization
            z = (best_seen - mean) / std
            ei = (best_seen - mean) * norm.cdf(z) + std * norm.pdf(z)
        else:  # maximization
            z = (mean - best_seen) / std
            ei = (mean - best_seen) * norm.cdf(z) + std * norm.pdf(z)

        return candidates[np.argmax(ei)]

