import random

import pandas as pd
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from pyDOE import lhs
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import os
import csv

from utils.logger import Logger


class MultiFidelityOptimizer:
    def __init__(self, workload_controller, fidelity_factors_info, evaluate_config_pop, target_system,
                 optimize_objective, max_iterations=200, fidelity_metric='kendall'):
        self.workload_controller = workload_controller
        self.fidelity_factors_info = fidelity_factors_info
        self.max_iterations = max_iterations
        self.target_system = target_system
        self.evaluate_config_pop = evaluate_config_pop
        self.evaluated_fidelities = set()
        self.consumed_budget_for_fidelity_search = 0
        self.budget_for_fidelity_search = 0
        self.fidelity_pop_size = 10
        self.fidelity_mutation_rate = 1 / int(len(self.fidelity_factors_info))
        self.fidelity_crossover_rate = 0.9
        self.fidelity_metric = fidelity_metric
        self.optimize_objective = optimize_objective
        self.logger = Logger(self.target_system, self.optimize_objective, self.workload_controller)

    def initialize_fidelity_pop(self, hf_factors, cost_related_factors, fidelity_pop_size):
        """
        Generate an initial fidelity population, make sure there are no duplicate fidelity combinations,
        and only change cost-related factors.
        :param hf_factors: default the highest fidelity factors
        :param cost_related_factors: a list that contains those cost-related factors
        :param fidelity_pop_size: population size of fidelity
        :return: initial fidelity population
        """
        fidelity_pop = []
        seen_fidelities = set()
        random_value = 0

        while len(fidelity_pop) < fidelity_pop_size:
            fidelity = hf_factors.copy()  # start from the highest fidelity
            for factor_name in cost_related_factors:
                factor_info = self.workload_controller.fidelity_factors_info[factor_name]
                if factor_info['type'] == 'integer':
                    random_value = random.randint(factor_info['min'], factor_info['max'])
                elif factor_info['type'] == 'float':
                    random_value = random.uniform(factor_info['min'], factor_info['max'])
                elif factor_info['type'] == 'enum':
                    possible_values = factor_info['enum_values']
                    random_value = random.choice(possible_values)
                # revise the value of cost-related fidelity factor
                fidelity[factor_name] = random_value

            fidelity_tuple = tuple(fidelity.items())
            if fidelity_tuple not in seen_fidelities:
                fidelity_pop.append(fidelity)
                seen_fidelities.add(fidelity_tuple)

        return fidelity_pop

    def evaluate_fidelity_pop(self, config_samples, fidelity_pop, hf_perf, rest_budget=None):
        """
        :param rest_budget:
        :param config_samples: a set of config for evaluating fidelity individual
        :param fidelity_pop: a fidelity pop to be evaluated
        :param hf_perf: the corresponding perf of configs_samples evaluated on high fidelity
        :return: evaluated_fidelity_pop: evaluated fidelity pop (format: [fidelity factors, corr, avg_cost])
        :return: cost_fidelity_pop: the total cost for evaluating the fidelity_pop
        """

        evaluated_fidelity_pop = []
        cost_fidelity_pop = 0
        for fidelity in fidelity_pop:
            if tuple(fidelity.items()) in self.evaluated_fidelities:
                continue

            lf_evaluated_samples, cost_fidelity_evaluation = self.evaluate_config_pop(config_samples, fidelity, stage_budget=rest_budget)
            lf_perf = [perf for _, perf, _ in lf_evaluated_samples]
            cost_fidelity_pop += cost_fidelity_evaluation
            self.consumed_budget_for_fidelity_search += cost_fidelity_evaluation

            # quantify the correlation between hf and lf, using kendall/spearman metric
            corr = 0
            if len(set(lf_perf)) <= 1:  # all perf are same (e.g., 0)
                print(f"[WARN]: Fidelity {fidelity} produced constant perf (e.g., all 0s). Setting correlation = -1")
                corr = -1
            else:
                try:
                    if self.fidelity_metric == 'kendall_corr':
                        corr, _ = kendalltau(hf_perf, lf_perf)
                    elif self.fidelity_metric == 'spearman_corr':
                        corr, _ = spearmanr(hf_perf, lf_perf)
                    else:
                        pass  # explore more metrics

                    # if still have nan, default as -1
                    if corr is None or pd.isna(corr):
                        print(f"[WARN] Correlation is NaN for fidelity {fidelity}. Resetting to -1.")
                        corr = -1
                except Exception as e:
                    print(f"Error when computing correlation for fidelity {fidelity}: {e}")
                    corr = -1

            # quantify the cost for evaluating a config under lf
            avg_cost = sum([cost for _, _, cost in lf_evaluated_samples]) / len(lf_evaluated_samples)
            evaluated_fidelity_pop.append((fidelity, corr, avg_cost))

            if self.consumed_budget_for_fidelity_search > self.budget_for_fidelity_search:
                break

        return evaluated_fidelity_pop, cost_fidelity_pop

    def evolutionary_search_fidelity(self, hf_factors, cost_related_factors, config_samples, hf_perf, log_path,
                                     cost_dva, budget_for_fidelity_search):

        # generate initial pop in fidelity space
        generation = 0
        self.consumed_budget_for_fidelity_search = cost_dva
        self.budget_for_fidelity_search = budget_for_fidelity_search

        init_fidelity_pop = self.initialize_fidelity_pop(hf_factors, cost_related_factors, self.fidelity_pop_size)
        evaluated_fidelity_pop, cost_fidelity_pop = self.evaluate_fidelity_pop(config_samples, init_fidelity_pop,
                                                                               hf_perf, rest_budget=self.budget_for_fidelity_search-self.consumed_budget_for_fidelity_search)

        # record those fidelities that have been evaluated
        for fidelity, _, _ in evaluated_fidelity_pop:
            self.evaluated_fidelities.add(tuple(fidelity.items()))

        evaluated_fidelity_pop, fidelity_pop_levels = self.nsga2_selection(evaluated_fidelity_pop,
                                                                           self.fidelity_pop_size)
        self.logger.store_fidelity_pop_to_csv(evaluated_fidelity_pop, fidelity_pop_levels, generation, log_path,
                                              self.fidelity_metric)

        # use MOEA to search promising fidelity individual
        while self.consumed_budget_for_fidelity_search < self.budget_for_fidelity_search:
            generation += 1
            exhausted = False
            if len(cost_related_factors) == 1:
                fidelity_offspring, exhausted = self.random_search_fidelity(evaluated_fidelity_pop,
                                                                            cost_related_factors[0],
                                                                            self.fidelity_pop_size)
            else:
                fidelity_offspring = []
                attempts = 0
                max_attempts = 100  # max times for generate new child (if generate 100 times child, still can not find a solution that have not been evaluated, stop exploration)

                while len(fidelity_offspring) < self.fidelity_pop_size and attempts < max_attempts:
                    parent1, parent2 = self.select_fidelity_parents(evaluated_fidelity_pop)
                    # child1, child2 = self.crossover_fidelity(parent1, parent2, cost_related_factors)
                    child1, child2 = self.uniform_crossover_fidelity(parent1, parent2, cost_related_factors)
                    child1 = self.mutate_fidelity(child1, cost_related_factors)
                    child2 = self.mutate_fidelity(child2, cost_related_factors)

                    # make sure the generated offspring are distinctive and unevaluated
                    child1_tuple = tuple(child1.items())
                    child2_tuple = tuple(child2.items())

                    added = False  # mark
                    if child1_tuple not in self.evaluated_fidelities and child1 not in fidelity_offspring:
                        fidelity_offspring.append(child1)
                        added = True

                    if len(fidelity_offspring) < self.fidelity_pop_size and child2_tuple not in self.evaluated_fidelities and child2 not in fidelity_offspring:
                        fidelity_offspring.append(child2)
                        added = True

                    if not added:
                        attempts += 1

                if attempts >= max_attempts:
                    print(
                        f"[Warn] Generation {generation}: no new fidelity offspring could be generated after multiple attempts. Marking as exhausted.")
                    exhausted = True

            evaluated_fidelity_offspring, cost_fidelity_pop = self.evaluate_fidelity_pop(config_samples,
                                                                                         fidelity_offspring, hf_perf, rest_budget=self.budget_for_fidelity_search-self.consumed_budget_for_fidelity_search)

            for fidelity, _, _ in evaluated_fidelity_offspring:
                self.evaluated_fidelities.add(tuple(fidelity.items()))

            combined_pop = evaluated_fidelity_pop + evaluated_fidelity_offspring

            # use NSGA-II to perform environmental selection
            evaluated_fidelity_pop, fidelity_pop_levels = self.nsga2_selection(combined_pop, self.fidelity_pop_size)
            self.logger.store_fidelity_pop_to_csv(evaluated_fidelity_pop, fidelity_pop_levels, generation, log_path,
                                                  self.fidelity_metric)

            if exhausted:
                break

        return evaluated_fidelity_pop

    def crossover_fidelity(self, parent1, parent2, cost_related_factors):

        """conduct partial match crossover （PMX） for fidelity factors """
        if random.random() > self.fidelity_crossover_rate:
            return parent1.copy(), parent2.copy()

        # select a subset of cost-related factors randomly for swap
        subset_size = random.randint(1, len(cost_related_factors))
        crossover_indices = random.sample(range(len(cost_related_factors)), subset_size)
        crossover_factors = [cost_related_factors[i] for i in crossover_indices]

        child1 = parent1.copy()
        child2 = parent2.copy()

        # exchange the value of crossover_factors
        for factor in crossover_factors:
            child1[factor], child2[factor] = parent2[factor], parent1[factor]

        return child1, child2

    def uniform_crossover_fidelity(self, parent1, parent2, cost_related_factors):
        """Conduct uniform crossover for fidelity factors."""

        # if not satisfy crossover condition, return parents
        if random.random() > self.fidelity_crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        # uniform crossover for each cost-related factor
        for factor in cost_related_factors:
            if random.random() > 0.5:
                child1[factor], child2[factor] = parent2[factor], parent1[factor]

        return child1, child2

    def mutate_fidelity(self, fidelity, cost_related_factors):

        """conduct mutation for fidelity factors/ make sure unique"""
        new_fidelity = fidelity.copy()
        new_value = 0
        # mutate for cost_related_factors
        for factor in cost_related_factors:
            if random.random() < self.fidelity_mutation_rate:
                factor_info = self.workload_controller.fidelity_factors_info[factor]
                if factor_info['type'] == 'integer':
                    new_value = random.randint(factor_info['min'], factor_info['max'])
                elif factor_info['type'] == 'float':
                    new_value = random.uniform(factor_info['min'], factor_info['max'])
                elif factor_info['type'] == 'enum':
                    possible_values = factor_info['enum_values']
                    new_value = random.choice(possible_values)
                new_fidelity[factor] = new_value

        # conduct mutation for cost-independent factors, with a small probability
        for other_factor in self.workload_controller.fidelity_factors_info.keys():
            if other_factor not in cost_related_factors and random.random() < self.fidelity_mutation_rate:
                factor_info = self.workload_controller.fidelity_factors_info[other_factor]
                if factor_info['type'] == 'integer':
                    new_fidelity[other_factor] = random.randint(factor_info['min'], factor_info['max'])
                elif factor_info['type'] == 'float':
                    new_fidelity[other_factor] = random.uniform(factor_info['min'], factor_info['max'])
                elif factor_info['type'] == 'enum':
                    possible_values = factor_info['enum_values']
                    new_fidelity[other_factor] = random.choice(possible_values)

        return new_fidelity

    def nsga2_selection(self, pop, pop_size):
        """
        Environmental selection: use NSGA-II to select pop_size individuals
        :param pop: population with objectives (factors, corr, cost)
        :param pop_size: population size
        :return: selected individuals and their respective front levels
        """
        # simulated_population = [
        #     ([random.randint(1, 5) for _ in range(5)],  # fidelity as a list [1, 2, 3, 4, 5]
        #      random.uniform(-1.0, 1.0),  # corr
        #      random.uniform(10, 100))  # cost
        #     for _ in range(20)
        # ]

        fronts = self.fast_non_dominated_sort(pop)
        new_pop = []
        pop_front_levels = []

        for i, front in enumerate(fronts):
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
                pop_front_levels.extend([i] * len(front))  # Store the front level for each individual
            else:
                # TODO: test the following code
                sorted_front = self.crowding_distance_sort(front)
                pop_front_levels.extend([i] * (pop_size - len(new_pop)))  # Store the front level for these individuals
                new_pop.extend(sorted_front[:pop_size - len(new_pop)])
                break

        return new_pop, pop_front_levels

    def fast_non_dominated_sort(self, pop):
        """
        fast non-dominated sort
        :param pop: fidelity population (factors, corr, cost)
        :return: fronts after sorting
        """
        fronts = [[]]
        domination_count = {}
        dominated_solutions = {}
        rank = {}

        for p in range(len(pop)):
            domination_count[p] = 0
            dominated_solutions[p] = []
            for q in range(len(pop)):
                result = self.dominates(pop[p], pop[q])
                if result == 1:
                    dominated_solutions[p].append(q)
                elif result == -1:
                    domination_count[p] += 1
            if domination_count[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        fronts.pop()
        sorted_fronts = [[pop[i] for i in front] for front in fronts]
        return sorted_fronts

    def crowding_distance_sort(self, front):
        """
        crowding distance sort
        :param front:
        :return:sorted front
        """
        if len(front) == 0:
            return front

        distances = [0.0] * len(front)
        num_objectives = 2

        for i in range(num_objectives):
            front.sort(key=lambda x: x[i + 1])
            distances[0] = distances[-1] = float('inf')
            for j in range(1, len(front) - 1):
                distances[j] += (front[j + 1][i + 1] - front[j - 1][i + 1]) / (front[-1][i + 1] - front[0][i + 1])

        sorted_front = [x for _, x in sorted(zip(distances, front), key=lambda pair: pair[0], reverse=True)]

        return sorted_front

    def dominates(self, individual1, individual2):
        """identify the domination relationship between individual1 and individual2"""
        corr1, cost1 = individual1[1], individual1[2]
        corr2, cost2 = individual2[1], individual2[2]

        if corr1 > corr2 and cost1 < cost2:
            return 1
        elif corr1 < corr2 and cost1 > cost2:
            return -1
        else:
            return 0

    def select_fidelity_parents(self, evaluated_pop):
        """binary selection, return two parents """

        def tournament(candidate1, candidate2):
            result = self.dominates(candidate1, candidate2)
            if result == 1:
                return candidate1[0]
            elif result == -1:
                return candidate2[0]
            else:
                return candidate1[0] if random.random() < 0.5 else candidate2[0]

        # select the first parent
        candidate1, candidate2 = random.sample(evaluated_pop, 2)
        parent1 = tournament(candidate1, candidate2)

        # select the second parent
        candidate3, candidate4 = random.sample(evaluated_pop, 2)
        parent2 = tournament(candidate3, candidate4)

        return parent1, parent2

    def decision_variable_analysis(self, hf_evaluated_samples, hf_factors, mt_sample=5, cd_threshold=5):

        """
        analysis which factors will significantly affect evaluation cost
        :param hf_evaluated_samples: sampled configs with objective perf evaluated on highest fidelity (config, perf, cost)
        :param hf_factors: default the highest fidelity setting
        :param mt_sample: max test sample, the upper limitation for testing individual factor
        :param cd_threshold: cost difference threshold, cost fluctuation surpass such threshold when revise one factor- > cost-related factor
        :return: cost-related factors (list)
        """

        def median_value(factor_info):
            if factor_info['type'] == 'integer':
                return (factor_info['min'] + factor_info['max']) // 2
            elif factor_info['type'] == 'float':
                return (factor_info['min'] + factor_info['max']) / 2
            elif factor_info['type'] == 'enum':
                values = factor_info['enum_values']
                return values[len(values) // 2]
            return None

        def normalize_config(config, knobs_info):
            normalized_config = {}
            for key, value in config.items():
                knob_info = knobs_info[key]
                if knob_info['type'] in ['integer', 'float']:
                    normalized_value = (value - knob_info['min']) / (knob_info['max'] - knob_info['min'])
                elif factor_info['type'] == 'enum':
                    normalized_value = knob_info['enum_values'].index(value) / (len(knob_info['enum_values']) - 1)
                normalized_config[key] = normalized_value
            return normalized_config

        def config_distance(config1, config2, knobs_info):
            norm_config1 = normalize_config(config1, knobs_info)
            norm_config2 = normalize_config(config2, knobs_info)
            distance = sum((norm_config1[key] - norm_config2[key]) ** 2 for key in norm_config1.keys())
            return distance ** 0.5

        cost_related_factors = []
        cost_dva = 0
        # (Yes/No/Uncertain)
        # Add the factors (that have been identified as cost related based on domain knowledge) to the list
        for factor_name, factor_info in self.workload_controller.fidelity_factors_info.items():
            if factor_info['cost_related'] == "Yes":
                cost_related_factors.append(factor_name)

        # Conduct dva only on factors whose properties we don't know
        for factor_name, factor_info in self.workload_controller.fidelity_factors_info.items():
            if factor_info['cost_related'] != "Uncertain":
                continue

            significant_difference = False

            # randomly select a config, then, select a config from the rest that most different from the current ones, max = 5
            selected_samples = [random.choice(hf_evaluated_samples)]
            for _ in range(1, mt_sample):
                remaining_samples = [sample for sample in hf_evaluated_samples if sample not in selected_samples]
                next_sample = max(remaining_samples, key=lambda sample: min(
                    config_distance(sample[0], prev_sample[0], self.target_system.knobs_info) for
                    prev_sample in selected_samples))
                selected_samples.append(next_sample)

            # sample -> config, perf, cost
            for selected_sample in selected_samples:
                selected_config = selected_sample[0]
                selected_hf_cost = selected_sample[2]

                # generate new low fidelity setting, only change one factor, extract the median of this dimension
                lf_factors = hf_factors.copy()
                lf_factors[factor_name] = median_value(factor_info)

                # evaluate sampled configs under lf settings (the func will return a pop format, but we only got one-> extract [0])
                lf_evaluated_sample, cost_selected_config = self.evaluate_config_pop([selected_config], lf_factors)
                lf_evaluated_sample = lf_evaluated_sample[0]
                lf_cost = lf_evaluated_sample[2]

                cost_dva += cost_selected_config
                # compare the cost of high and low fidelity settings
                cost_difference = abs(lf_cost - selected_hf_cost)
                cost_change_percentage = (cost_difference / selected_hf_cost) * 100

                if cost_change_percentage > cd_threshold:  # justify whether this factor is a cost-related factor
                    significant_difference = True
                    cost_related_factors.append(factor_name)
                    break

            if significant_difference:
                continue

        return cost_related_factors, cost_dva

    def random_search_fidelity(self, evaluated_fidelity_pop, cost_related_factor_name, fidelity_pop_size):

        """random search, if there is only one cost_related_factor"""
        fidelity_offspring = []
        seen_fidelities = set([tuple(fidelity.items()) for fidelity, _, _ in evaluated_fidelity_pop])
        factor_info = self.workload_controller.fidelity_factors_info[cost_related_factor_name]

        # get all of possible values of the cost_related_factor_name
        if factor_info['type'] == 'integer':
            possible_values = list(range(factor_info['min'], factor_info['max'] + 1))
        elif factor_info['type'] == 'float':
            possible_values = None  # as for float type, without limited possible values 对于 float
        elif factor_info['type'] == 'enum':
            possible_values = factor_info['enum_values']

        exhausted = False  # mark whether the fidelity space has been fully explored
        while len(fidelity_offspring) < fidelity_pop_size:
            if factor_info['type'] in ['integer', 'enum']:
                # whether all the possible values have been explored ? as for integer and enum
                remaining_values = [val for val in possible_values if tuple({**evaluated_fidelity_pop[0][0],
                                                                             cost_related_factor_name: val}.items()) not in seen_fidelities]
                if not remaining_values:
                    print(f"All possible values for {cost_related_factor_name} have been explored.")
                    exhausted = True
                    break

                random_value = random.choice(remaining_values)
            elif factor_info['type'] == 'float':
                # whether all the possible values have been explored ? as for float
                random_value = round(random.uniform(factor_info['min'], factor_info['max']), 2)
                fidelity_tuple = tuple(
                    {**evaluated_fidelity_pop[0][0], cost_related_factor_name: random_value}.items())

                # make sure the generated value has not been explored
                while fidelity_tuple in seen_fidelities:
                    random_value = round(random.uniform(factor_info['min'], factor_info['max']), 2)
                    fidelity_tuple = tuple(
                        {**evaluated_fidelity_pop[0][0], cost_related_factor_name: random_value}.items())

            fidelity = evaluated_fidelity_pop[0][0].copy()
            fidelity[cost_related_factor_name] = random_value

            fidelity_tuple = tuple(fidelity.items())

            # make sure the new generated fidelity individual has not been explored or exists in current offspring
            if fidelity_tuple not in seen_fidelities:
                fidelity_offspring.append(fidelity)
                seen_fidelities.add(fidelity_tuple)

                # mutate the other cost-independent factors with a small probability
                for other_factor_name in self.workload_controller.fidelity_factors_info.keys():
                    if other_factor_name != cost_related_factor_name and random.random() < self.fidelity_mutation_rate:
                        other_factor_info = self.workload_controller.fidelity_factors_info[other_factor_name]
                        if other_factor_info['type'] == 'integer':
                            fidelity[other_factor_name] = random.randint(other_factor_info['min'],
                                                                         other_factor_info['max'])
                        elif other_factor_info['type'] == 'float':
                            fidelity[other_factor_name] = random.uniform(other_factor_info['min'],
                                                                         other_factor_info['max'])
                        elif other_factor_info['type'] == 'enum':
                            possible_values = other_factor_info['enum_values']
                            fidelity[other_factor_name] = random.choice(possible_values)

        return fidelity_offspring, exhausted

    def select_fidelity_for_stages_dbscan(self, optimized_fidelity_pop, log_path, num_stages=None):
        """
        :param log_path:
        :param optimized_fidelity_pop, each ind: [fidelity_factors, fidelity_value, cost]
        :param num_stages: The number of stages
        :return: selected fidelities and corresponding accuracy (list)
        """

        # TODO: if the fidelities are evenly distributed, specific operation should be induced
        # conduct non-dominated sorting for optimized_fidelity_pop
        fronts = self.fast_non_dominated_sort(optimized_fidelity_pop)
        first_front = fronts[0]

        # filter those fidelity settings that with low accuracy, e.g., fidelity <= 0.39
        first_front = [ind for ind in first_front if ind[1] > 0.39]
        if not first_front:
            print("[WARN] all the accuracy of fidelity settings are lower than 0.39, only using high-fidelity instead")
            return None

        first_front.sort(key=lambda x: x[1])

        if len(first_front) == 1:
            self.logger.log_fidelity_ind_clusters(first_front, [0], log_path)
            print(f"[3] Only one non-dominated fidelity setting")
            return first_front

        # obtain max and min fidelity settings
        min_fidelity_ind = min(first_front, key=lambda x: x[1])
        max_fidelity_ind = max(first_front, key=lambda x: x[1])

        corrs = np.array([ind[1] for ind in first_front]).reshape(-1, 1)

        # calculate the distance far from most close neighbor for all ind, and their average is defined as radius for clustering
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(corrs)
        distances, indices = nbrs.kneighbors(corrs)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        eps = np.mean(distances)
        db = DBSCAN(eps=eps, min_samples=2).fit(corrs)
        labels = db.labels_

        # check whether min and max fidelity settings are divided into the same cluster （scatter point are labelled as -1）
        min_label = labels[first_front.index(min_fidelity_ind)]
        max_label = labels[first_front.index(max_fidelity_ind)]
        if min_label == max_label:
            # if so, split it
            labels[first_front.index(max_fidelity_ind)] = -2

        self.logger.log_fidelity_ind_clusters(first_front, labels, log_path)
        print(f"[3] Divide fidelity settings are recorded in {log_path}")

        # clustering those inds with same label
        clusters = {}
        for label, individual in zip(labels, first_front):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(individual)

        # select the ind with the smallest cost from each cluster
        selected_fidelities = [min(cluster, key=lambda x: x[2]) for cluster in clusters.values()]
        selected_fidelities.sort(key=lambda x: x[1])

        # if num_stages was defined and the num of cluster smaller than num_stages
        if num_stages:
            while len(selected_fidelities) < num_stages:
                print("[WARN] Non-dominated fidelity settings less than defined stages")
                selected_fidelities.append(selected_fidelities[-1])
            return selected_fidelities[:num_stages]
        else:
            return selected_fidelities


    def select_fidelity_by_knee_point(self, optimized_fidelity_pop, log_path, num_stages=None):
        """
        Select fidelity setting based on the knee point from the non-dominated front.
        :param optimized_fidelity_pop: List of individuals [fidelity_factors, fidelity_value, cost]
        :param log_path: For logging
        :param num_stages: Kept for interface consistency, but only 1 point is selected now
        :return: selected fidelity setting(s) as a list
        """

        fronts = self.fast_non_dominated_sort(optimized_fidelity_pop)
        first_front = fronts[0]
        first_front = [ind for ind in first_front if ind[1] > 0.39]

        if not first_front:
            print("[WARN] all the accuracy of fidelity settings are lower than 0.39, using high-fidelity instead")
            return None

        if len(first_front) == 1:
            self.logger.log_fidelity_ind_clusters(first_front, [0], log_path)
            print("[KNEE] Only one non-dominated fidelity setting")
            return first_front

        # 只根据corr/cost 来选择fidelity
        if len(first_front) >= 2:
            best = max(first_front, key=lambda x: x[1] / x[2])  # max corr / cost
            self.logger.log_fidelity_ind_clusters([best], [0], log_path)
            print(f"[KNEE] {len(first_front)} points, selected by max corr/cost")
            return [best]

        # Normalize for knee detection (corr: y, cost: x)
        points = np.array([[ind[2], ind[1]] for ind in first_front])  # [cost, corr]
        sorted_idx = np.argsort(points[:, 0])
        points = points[sorted_idx]
        sorted_front = [first_front[i] for i in sorted_idx]

        # Define line between first and last
        p1, p2 = points[0], points[-1]
        line_vec = p2 - p1
        line_vec_norm = line_vec / np.linalg.norm(line_vec)

        # Calculate orthogonal distance from each point to the line
        vecs = points - p1
        proj = np.dot(vecs, line_vec_norm)[:, None] * line_vec_norm
        dists = np.linalg.norm(vecs - proj, axis=1)

        knee_index = int(np.argmax(dists))
        selected_fidelities = [sorted_front[knee_index]]

        self.logger.log_fidelity_ind_clusters(selected_fidelities, [0], log_path)
        print(f"[KNEE] Knee point selected with accuracy={selected_fidelities[0][1]:.4f}, cost={selected_fidelities[0][2]:.4f}")

        return selected_fidelities


