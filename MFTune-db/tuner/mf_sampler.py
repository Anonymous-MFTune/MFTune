import csv
import os
import time
import itertools
import numpy as np
from pyDOE import lhs
from systems.mysqldb import MysqlDB
from systems.postgresqldb import PostgresqlDB
from workload import WorkloadController


class MFSampler:
    def __init__(self, args_db, args_workload, args_tune, run):
        super().__init__()
        self.args_db = args_db
        self.args_workload = args_workload
        self.args_tune = args_tune
        self.sys_name = self.args_db['db']
        self.target_system = self.get_target_system()
        self.workload_controller = WorkloadController(args_db, args_workload, self.target_system)

        self.sample_size = 1000
        self.data_dir = os.path.join('sampling_results', f'{self.sys_name}')
        os.makedirs(self.data_dir, exist_ok=True)

        self.config_samples_path = os.path.join(self.data_dir, f'{self.sys_name}_lhs_configs.csv')
        self.fidelity_file_path = os.path.join('sampling_results', f'{self.sys_name}', f'{self.sys_name}_fidelities.csv')
        self.log_path = os.path.join('sampling_results', self.sys_name, 'evaluated_configs', f'run_{run}')

        os.makedirs(os.path.dirname(self.config_samples_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.fidelity_file_path), exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

    def get_target_system(self):
        """initialize db """
        if self.sys_name == 'mysql':
            return MysqlDB(self.args_db)
        elif self.sys_name == 'postgresql':
            return PostgresqlDB(self.args_db)
        raise ValueError(f"Unsupported database type: {self.sys_name}")

    def sampling_and_evaluate(self):
        """sampling configs，generate fidelities，evaluate configs under each fidelities"""
        configs = self.sample_configs_by_lhs(self.sample_size)
        fidelities = self.read_or_generate_fidelities()

        for fidelity in fidelities:
            log_file = self.generate_data_file_name(fidelity)
            log_file_path = os.path.join(self.log_path, log_file)

            existing_configs = self.read_existing_configs(log_file_path)

            for config in configs:
                config_tuple = tuple(config.items())
                if config_tuple in existing_configs:
                    continue

                self.evaluate_config(config, fidelity, log_file_path)

    def evaluate_config(self, config, fidelity, log_file_path):
        """evaluate a config under specified fidelity """
        print(f"Set the config to {self.sys_name}: {config}")
        self.target_system.set_db_knob(config)

        start_time = time.time()
        try:
            print(f"Execute workload: {fidelity}")
            latency, throughput, prepare_time, run_time, clean_time = self.workload_controller.run_workload(
                fidelity)
        except Exception as e:
            print(f"Workload execution failed: {e}")
            latency = throughput = prepare_time = run_time = clean_time = 0
        evaluated_time = time.time() - start_time

        print(f"[PERFORMANCE]: Throughput: {throughput}")
        print("-------------------------------------------------")

        self.logging_data(log_file_path, config, latency, throughput, evaluated_time, prepare_time, run_time,
                          clean_time)

    def sample_configs_by_lhs(self, num_samples):
        """ sampling configs by using LHS and save to .csv """
        if os.path.exists(self.config_samples_path):
            existing_configs = self.read_csv_configs(self.config_samples_path)
            if len(existing_configs) >= num_samples:
                return existing_configs

            additional_samples = num_samples - len(existing_configs)
        else:
            existing_configs = []
            additional_samples = num_samples

        num_params = len(self.target_system.knobs_info)
        lhs_sample = lhs(num_params, samples=additional_samples)

        new_configs = []
        for i in range(additional_samples):
            config = {}
            for j, (key, val) in enumerate(self.target_system.knobs_info.items()):
                if val['type'] == 'integer':
                    range_width = val['max'] - val['min'] + 1
                    config[key] = int(lhs_sample[i][j] * range_width) + val['min']
                elif val['type'] == 'float':
                    range_width = val['max'] - val['min']
                    config[key] = lhs_sample[i][j] * range_width + val['min']
                elif val['type'] == 'enum':
                    possible_values = val['enum_values']
                    config[key] = possible_values[int(lhs_sample[i][j] * len(possible_values))]
            new_configs.append(config)

        all_configs = existing_configs + new_configs
        self.write_csv_configs(all_configs, self.config_samples_path)
        return all_configs

    def read_or_generate_fidelities(self):
        """read fidelity file, if exists , otherwise generate fidelity settings and save to .csv """
        if os.path.exists(self.fidelity_file_path):
            return self.read_csv_fidelities(self.fidelity_file_path)

        fidelities = self.generate_fidelity_samples()
        self.write_csv_fidelities(fidelities)
        return fidelities

    def generate_fidelity_samples(self):
        fidelity_factors = self.workload_controller.fidelity_factors_info
        fidelity_keys = list(fidelity_factors.keys())
        fidelity_values = [factor['enum_values'] for factor in fidelity_factors.values()]
        return [dict(zip(fidelity_keys, values)) for values in itertools.product(*fidelity_values)]

    @staticmethod
    def read_csv_configs(file_path):
        """read CSV for extracting configs"""
        with open(file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            return [row for row in reader]

    @staticmethod
    def read_csv_fidelities(file_path):
        """read CSV for extracting fidelities"""
        with open(file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            return [dict(row) for row in reader]

    def write_csv_configs(self, configs, file_path):
        """write configs to CSV file"""
        with open(file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(self.target_system.knobs_info.keys()))
            writer.writeheader()
            writer.writerows(configs)

    def write_csv_fidelities(self, fidelities):
        """read fidelities to CSV file"""
        with open(self.fidelity_file_path, 'w', newline='') as file:
            fieldnames = list(fidelities[0].keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fidelities)

    def read_existing_configs(self, file_path):

        existing_configs = set()
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader, None)  # skip header
                for row in reader:
                    config_keys = list(self.target_system.knobs_info.keys())
                    config_values = row[:len(config_keys)]
                    existing_configs.add(tuple(zip(config_keys, config_values)))
        return existing_configs

    def logging_data(self, log_file_path, config, latency, throughput, evaluated_time, prepare_time, run_time,
                     clean_time):
        """recording configs and corresponding perfs """
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                header = list(self.target_system.knobs_info.keys()) + ['latency', 'throughput', 'evaluated_time',
                                                                       'prepare_time', 'run_time', 'clean_time']
                writer.writerow(header)

        with open(log_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            row = [config[param] for param in self.target_system.knobs_info.keys()] + [latency, throughput,
                                                                                       evaluated_time, prepare_time,
                                                                                       run_time, clean_time]
            writer.writerow(row)

    @staticmethod
    def generate_data_file_name(factors):
        """generate file name"""
        return f"{factors['time']}_runtime_{factors['tables']}_tbls_{factors['table-size']}_tab_size_{factors['threads']}_threads_{factors['r_ratio']}_rw.csv"
