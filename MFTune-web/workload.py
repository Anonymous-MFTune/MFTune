import json
import random
import subprocess
import time
import shlex
import os
import pandas as pd
import re
import xml.etree.ElementTree as ET
import threading


class WorkloadController:
    def __init__(self, args_server, args_workload, target_system):
        self.server_url = args_server['url']
        self.port = args_server['port']
        self.container_name = args_server['container_name']

        self.workload_bench = args_workload['workload_bench']
        self.sys_name = args_server['server']
        self.target_system = target_system

        # Fidelity Factors Info
        self.fidelity_factors_info = self.initialize_fidelity_factors(args_workload['fidelity_factor_file'],
                                                                      int(args_workload['fidelity_factor_num']))
        self.default_fidelity_factors = self.get_default_fidelity_factors()

    # Initialize fidelity_factors, including the name, type, value and so on.
    @staticmethod
    def initialize_fidelity_factors(fidelity_factor_file, fidelity_factor_num):
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

    # Get default fidelity factors, which is considered as the highest fidelity, original task, i.e., true evaluation
    @staticmethod
    def get_default_fidelity_factors():
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

        RPS, TPR = 0, 0
        if self.workload_bench == 'ab':
            RPS, TPR = self.run_ab(factors)
        elif self.workload_bench == 'wrk':
            RPS, TPR = self.run_wrk(factors)
        elif self.workload_bench == '***':
            pass

        return float(RPS), float(TPR)

    def run_ab(self, factors):

        # requests = factors["requests"]
        # concurrency = factors["concurrency"]
        # timelimit = factors["timelimit"]
        # post = factors["post"]
        #
        #
        # if post:
        #     command = f"ab -n {requests} -c {concurrency} -t {timelimit} -p post_data.txt -T \"application/x-www-form-urlencoded\" {self.server_url}posttest"
        # else:
        #     command = f"ab -n {requests} -c {concurrency} -t {timelimit} {self.server_url}"

        command = self.generate_ab_command(factors, self.server_url)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"Executing workload [{factors}]...")
                result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
                output = result.stdout
                if "error" not in output.lower():
                    RPS, TPR = self.evaluate_performance_ab(output)
                    print(f"[PERFORMANCE]: RPS: {RPS} | TPR: {TPR}")
                    print("-----------------------------------------------------")
                    return float(RPS), float(TPR)
                else:
                    print(f"[Attempt {attempt + 1}]: error during benchmark run: {output}")
            except subprocess.CalledProcessError as e:
                print(f"[Attempt {attempt +1}]: error during benchmark run: {e}")
                output = "error"
            time.sleep(1)

        # fail after multiple tries, return default value 0, 0 for [RPS, TPR]
        print("All benchmark attempts failed. Returning default performance (0, 0).")
        return 0, 0

    @staticmethod
    def evaluate_performance_ab(output):
        requests_per_second_pattern = r"Requests per second:\s+(\d+\.\d+)"
        time_per_request_pattern = r"Time per request:\s+(\d+\.\d+)"
        requests_per_second = re.search(requests_per_second_pattern, output).group(1)
        time_per_request = re.search(time_per_request_pattern, output).group(1)
        return float(requests_per_second), float(time_per_request)

    @staticmethod
    def generate_ab_command(factors, server_url):
        """Generate ab (ApacheBench) command dynamically based on provided factors."""
        command_parts = ["ab"]

        # mapping selectable factors to their identifiers in command line.
        param_flags = {
            "requests": "-n",
            "concurrency": "-c",
            "timelimit": "-t",
        }

        for key, flag in param_flags.items():
            if key in factors:
                command_parts.extend([flag, str(factors[key])])

        if factors.get("post", False):
            command_parts.extend(["-p", "post_data.txt", "-T", "application/x-www-form-urlencoded"])
            command_parts.append(f"{server_url}")  # posttest
        else:
            command_parts.append(server_url)

        return " ".join(command_parts)

    def run_wrk(self, factors):
        command = self.generate_wrk_command(factors, self.server_url)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"Executing wrk workload [{factors}]...")
                result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
                output = result.stdout
                if "error" not in output.lower():
                    RPS, LAT = self.evaluate_performance_wrk(output)
                    print(f"[PERFORMANCE]: RPS: {RPS} | LAT: {LAT}")
                    print("-----------------------------------------------------")
                    return float(RPS), float(LAT)
                else:
                    print(f"[Attempt {attempt + 1}]: error during benchmark run: {output}")
            except subprocess.CalledProcessError as e:
                print(f"[Attempt {attempt + 1}]: error during wrk run: {e}")
            time.sleep(1)

        print("All wrk attempts failed. Returning default performance (0, 0).")
        return 0, 0

    @staticmethod
    def evaluate_performance_wrk(output):
        rps_pattern = r"Requests/sec:\s+(\d+\.\d+)"
        latency_pattern = r"Latency\s+(\d+\.\d+)"
        rps = re.search(rps_pattern, output).group(1)
        latency = re.search(latency_pattern, output).group(1)
        return float(rps), float(latency)

    @staticmethod
    def generate_wrk_command(factors, server_url):
        command_parts = ["wrk"]

        if "threads" in factors:
            command_parts.extend(["-t", str(factors["threads"])])
        if "connections" in factors:
            command_parts.extend(["-c", str(factors["connections"])])
        if "duration" in factors:
            command_parts.extend(["-d", f"{factors['duration']}s"])

        if factors.get("post", False):
            command_parts.extend(["-s", "post.lua"])  # self defined wrk Lua script
            command_parts.append(f"{server_url}")
        else:
            command_parts.append(server_url)

        return " ".join(command_parts)
