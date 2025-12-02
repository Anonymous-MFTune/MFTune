import json
import subprocess
import time
import shlex
import os
import pandas as pd
from utils.db_connector import DBConnector
from utils.db_connector import MysqlConnector
import re
import xml.etree.ElementTree as ET
import random


class WorkloadController:
    def __init__(self, args_db, args_workload, target_system):
        self.error = None
        self.output = None
        self.workload_bench = args_workload['workload_bench']
        self.host = args_db['host']
        self.user = args_db['user']
        self.port = args_db['port']
        self.password = args_db['password']
        self.dbname = args_workload['dbname']
        self.sys_name = args_db['db']
        if args_workload['lua_path'] is not None:
            self.lua_path = args_workload['lua_path']
        self.target_system = target_system

        # Fidelity Factors Info
        self.fidelity_factors_info = self.initialize_fidelity_factors(args_workload['fidelity_factor_file'],
                                                                      int(args_workload['fidelity_factor_num']))
        self.default_fidelity_factors = self.get_default_fidelity_factors()


    @staticmethod
    def initialize_fidelity_factors(fidelity_factor_file, fidelity_factor_num):
        """
        Initialize fidelity_factors, including the name, type, value and so on.
        """
        global FIDELITY_FACTORS
        global FIDELITY_FACTORS_INFO
        if fidelity_factor_num == -1:
            f = open(fidelity_factor_file)
            FIDELITY_FACTORS_INFO = json.load(f)
            FIDELITY_FACTORS = list(FIDELITY_FACTORS_INFO.keys())
            f.close()
        else:
            f = open(fidelity_factor_file)
            factor_tmp = json.load(f)
            i = 0
            FIDELITY_FACTORS_INFO = {}
            while i < fidelity_factor_num:
                key = list(factor_tmp.keys())[i]
                FIDELITY_FACTORS_INFO[key] = factor_tmp[key]
                i = i + 1
            FIDELITY_FACTORS = list(FIDELITY_FACTORS_INFO.keys())
            f.close()
        return FIDELITY_FACTORS_INFO


    @staticmethod
    def get_default_fidelity_factors():
        """
        Get default fidelity factors, which is considered as the highest fidelity, original task, i.e., true evaluation
        """
        default_factors = {}
        for name, value in FIDELITY_FACTORS_INFO.items():
            if not value['type'] == "combination":
                default_factors[name] = value['default']
            else:
                pass
        return default_factors

    @staticmethod
    def get_random_fidelity_factors(exclude_default: bool = False):
        """
        randomly generate a fidelity setting in the fidelity space
        currently enum/categorical is supported
        other types (int/float/bool) are reserved for future implementation
        """
        rnd = {}
        for name, info in FIDELITY_FACTORS_INFO.items():
            ftype = str(info.get("type", "")).lower()

            if ftype == "combination":
                continue

            if ftype in ("enum", "categorical") or ("enum_values" in info or "values" in info):
                values = info.get("enum_values") or info.get("values") or []
                if not values:
                    continue
                if exclude_default and "default" in info and len(values) > 1:
                    choices = [v for v in values if v != info["default"]] or values
                else:
                    choices = values
                rnd[name] = random.choice(choices)
                continue

            if ftype in ("int", "integer"):
                pass
            elif ftype == "float":
                pass
            elif ftype in ("bool", "boolean"):
                pass
            else:
                if "default" in info:
                    rnd[name] = info["default"]

        return rnd


    def run_workload(self, factors):

        latency, throughput, prepare_time, run_time, clean_time = 0, 0, 0, 0, 0
        if self.workload_bench == 'sysbench':
            latency, throughput, prepare_time, run_time, clean_time = self.run_sysbench(factors)
        elif self.workload_bench == 'tpcc':
            latency, throughput, prepare_time, run_time, clean_time = self.run_tpcc(factors)
        elif self.workload_bench == 'ycsb':
            pass

        return float(latency), float(throughput), float(prepare_time), float(run_time), float(clean_time)

    def run_sysbench(self, factors):

        """
        :param factors: fidelity factors (dic), e.g., tables: 6, table-size...
        :return:
        """
        # --db-driver (default as database)
        cmd_prefix = ''
        if self.sys_name == 'mysql':
            cmd_prefix = f"sysbench {self.lua_path} --mysql-host={self.host} --mysql-user={self.user} --mysql-password={self.password} --mysql-db={self.dbname}"
        elif self.sys_name == 'postgresql':
            cmd_prefix = f"sysbench {self.lua_path} --db-driver=pgsql --pgsql-host={self.host} --pgsql-port={self.port} --pgsql-user={self.user} --pgsql-password={self.password} --pgsql-db={self.dbname} --auto-inc=true"

        # dynamically set the fidelity factors (e.g., events,tables, table_size);
        for key, value in factors.items():
            if key == "r_ratio":  # r_ratio cannot be set here, properly be set by using lua scripts
                if value == 0.5:
                    cmd_prefix += f" --point-selects={0}"
                elif value == 0.6:
                    cmd_prefix += f" --point-selects={2}"
                elif value == 0.7:
                    cmd_prefix += f" --point-selects={5}"
                elif value == 0.8:
                    cmd_prefix += f" --point-selects={12}"
                elif value == 0.9:
                    cmd_prefix += f" --point-selects={32}"
            else:
                cmd_prefix += f" --{key}={value}"

        # Set env parameter for read_ratio/ ignore this, as we can use point-selects to control r_ratio
        # os.environ["READ_RATIO"] = str(factors.get("r_ratio", 0.5))

        # Delete the old db if exist and create a new db
        self.target_system.manage_database("drop", f"{self.dbname}")
        time.sleep(3)
        self.target_system.manage_database("create", f"{self.dbname}")

        # Prepare data
        start_time = time.time()
        cmd_prepare = cmd_prefix + " prepare"
        subprocess.call(cmd_prepare, shell=True)
        end_time = time.time()
        prepare_time = end_time - start_time

        # Run the workload
        start_time = time.time()
        cmd_run = cmd_prefix + " run"
        # output = subprocess.getoutput(cmd_run)
        cmd_list = shlex.split(cmd_run)
        process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.output, self.error = process.communicate()
        end_time = time.time()
        run_time = end_time - start_time

        # Get the corresponding perf, e.g., throughput/latency
        latency, throughput = self.evaluate_performance_sysbench()

        # Clean data
        start_time = time.time()
        cmd_cleanup = cmd_prefix + " cleanup"
        subprocess.call(cmd_cleanup, shell=True)
        end_time = time.time()
        clean_time = end_time - start_time

        return float(latency), float(throughput), float(prepare_time), float(run_time), float(clean_time)

    def evaluate_performance_sysbench(self):
        """evaluate the performance of current configs"""
        avg_latency = 0
        transactions_per_sec = 0

        # transfer the stream as line list
        lines = self.output.decode('utf-8').splitlines()

        print("==== SYSBENCH OUTPUT ====")
        print(self.output.decode("utf-8"))
        print("=========================")

        for line in lines:
            if 'min:' in line:
                min_latency = float(line.split(':')[1].strip())
            elif 'avg:' in line:
                avg_latency = float(line.split(':')[1].strip())
            elif 'max:' in line:
                max_latency = float(line.split(':')[1].strip())
            elif '95th percentile:' in line:
                percentile_95 = float(line.split(':')[1].strip())
            elif 'total time:' in line:
                total_time = float(line.split(':')[1].strip()[:-1])  # Remove 's'
            elif 'total number of events:' in line:
                total_events = int(line.split(':')[1].strip())
            elif 'transactions:' in line:
                transactions_per_sec = float(line.split('(')[1].split(' ')[0])
            elif 'queries:' in line and '(per sec.)' in line:
                queries_per_sec = float(line.split('(')[1].split(' ')[0])
            elif "read:" in line:
                read_ops = int(re.search(r'read:\s+(\d+)', line).group(1))
            elif "write:" in line:
                write_ops = int(re.search(r'write:\s+(\d+)', line).group(1))

        return avg_latency, transactions_per_sec

    def set_fidelity_factors(self, config_file, factors):
        """
        update the parameter of OLTPBench file (e.g., scale factor, time)
        :param config_file: file path
        :param factors: dic with fidelity-key and fidelity-value (e.g., scalefactor, time)
        """
        tree = ET.parse(config_file)
        root = tree.getroot()

        # set the scale factor and time dynamically (control the fidelity)
        for key, value in factors.items():
            for elem in root.iter(key):
                elem.text = str(value)

        # update the connection info of db
        for elem in root.iter('DBUrl'):
            elem.text = f"jdbc:{self.sys_name}://{self.host}:{self.port}/{self.dbname}"
        for elem in root.iter('username'):
            elem.text = self.user
        for elem in root.iter('password'):
            elem.text = self.password

        # save the updated config file
        tree.write(config_file)
        print(f"Updated {config_file} with factors: {factors}")

    def run_tpcc(self, factors):
        """
        Run TPCC benchmark test
        :param factors: fidelity factors (dic), which is using for adjust workload
        current tpcc factors include time and scale factor, where scale factor control the size of data (1->100M)
        :return: latency, throughput, prepare_time, run_time, clean_time
        """

        # delete db and recreate db, ensuring the environment is clean
        self.target_system.manage_database("drop", f"{self.dbname}")
        self.target_system.manage_database("create", f"{self.dbname}")
        config_file = None
        if self.sys_name == "mysql":
            config_file = "workload/oltpbench/config/tpcc_config_mysql.xml"
        elif self.sys_name == "postgresql":
            config_file = "workload/oltpbench/config/tpcc_config_postgres.xml"

        # update scale factor and time (change workload/fidelity) in config_file/xml
        self.set_fidelity_factors(config_file, factors)

        # construct OLTPBench cmd_prefix (as a list)
        cmd_prefix = [
            "java",
            "-cp", "workload/oltpbench/target/oltpbench-1.0-jar-with-dependencies.jar:workload/oltpbench/lib/*",
            "-Dlog4j.configuration=workload/oltpbench/log4j.properties",
            "com.oltpbenchmark.DBWorkload",
            "-b", "tpcc",
            "-c", config_file,
            "-o", "tpcc_result",  # output file name, under results directory
            "-s", "5"  # inspecting data per 5s
        ]

        # prepare
        start_time = time.time()
        cmd_prepare = cmd_prefix + ["--create=true", "--load=true"]
        subprocess.call(cmd_prepare)
        end_time = time.time()
        prepare_time = end_time - start_time

        # run
        start_time = time.time()
        cmd_run = cmd_prefix + ["--execute=true"]
        process = subprocess.Popen(cmd_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # the output represents the results that obtained by averaging the results from multiple inspection
        # e.g., per 5s, aggreating the results and averaging the results
        self.output, self.error = process.communicate()
        end_time = time.time()
        run_time = end_time - start_time

        latency, throughput = self.evaluate_performance_tpcc()

        # clean up; due to the command lines are different from sysbench, we ignore this operation
        start_time = time.time()
        # cmd_cleanup = cmd_prefix + ["--cleanup=true"]
        # subprocess.call(cmd_cleanup)
        end_time = time.time()
        clean_time = end_time - start_time

        return float(latency), float(throughput), float(prepare_time), float(run_time), float(clean_time)

    @staticmethod
    def evaluate_performance_tpcc():

        res_file = "results/tpcc_result.res"  # file name is like tpcc_result.res
        csv_file = "results/tpcc_result.csv"

        if not os.path.exists(res_file):
            raise FileNotFoundError(f"{res_file} does not exist.")

        df = pd.read_csv(res_file)
        df.columns = df.columns.str.strip()  # remove the space in front of the column
        # print("Columns after cleaning:", list(df.columns))

        # calculate ave throughput and 95th avg latency (avg_lat)
        avg_throughput = df['throughput(req/sec)'].mean() if 'throughput(req/sec)' in df.columns else None
        avg_95th_latency = df["95th_lat(ms)"].mean()

        # delete .res file
        os.remove(res_file)
        os.remove(csv_file)

        return avg_95th_latency, avg_throughput

