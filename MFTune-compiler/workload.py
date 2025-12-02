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
from subprocess import CalledProcessError, TimeoutExpired


class WorkloadController:
    def __init__(self, args_compiler, args_workload, target_system):

        self.port = args_compiler['port']
        self.container_name = args_compiler['container_name']
        self.workload_bench = args_workload['workload_bench']
        self.sys_name = args_compiler['compiler']
        self.target_system = target_system
        self.output_file = 'test.c'
        self.seed = 3714764995
        # self.seed = random.randint(1, 2**32 - 1)
        self.include_file = '/usr/local/include'

        # Fidelity Factors Info
        self.fidelity_factors_info = self.initialize_fidelity_factors(args_workload['fidelity_factor_file'],
                                                                      int(args_workload['fidelity_factor_num']))
        self.default_fidelity_factors = self.get_default_fidelity_factors()

    @staticmethod
    def initialize_fidelity_factors(fidelity_factor_file, fidelity_factor_num):
        # Initialize fidelity_factors, including the name, type, value and so on.
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
        # Get default fidelity factors, which is considered as the highest fidelity, original task, i.e., true evaluation
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
    
    def run_workload(self, factors, config):

        run_time, compile_time, exe_size, csmith_time, source_size, source_lines = 999999999, 999999999, 999999999, 999999999, 999999999, 999999999
        if self.workload_bench == 'csmith':
            run_time, compile_time, exe_size, csmith_time, source_size, source_lines = self.run_csmith(factors, config)
        elif self.workload_bench == '***':
            pass

        return run_time, compile_time, exe_size, csmith_time, source_size, source_lines

    def run_csmith(self, factors, config):

        # TODO: test whether cloc... will consume a part of cost
        # prepare workload (generate .c file)
        benchmark_cmd = self.generate_csmith_command(factors)
        try:
            # ==================prepare stage====================
            # run benchmark_cmd in container for generating .c file
            print(f"[RUN BENCHMARK CMD] | {benchmark_cmd}")
            start_csmith = time.time()
            self.run_cmd_in_container(benchmark_cmd, timeout=300)
            end_csmith = time.time()
            csmith_time = end_csmith - start_csmith

            # Print the generated C code
            # print("[GENERATED C CODE] ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
            # cat_cmd = ["cat", self.output_file]
            # stdout, _ = self.run_cmd_in_container(cat_cmd)
            # print(stdout)
            # print("↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")

            # get the size of generated .c file (source code)
            size_cmd = ["stat", "-c", "%s", self.output_file]
            stdout, _ = self.run_cmd_in_container(size_cmd)
            source_size = int(stdout.strip())

            # calculate the LOC using cloc
            cloc_cmd = ["cloc", self.output_file, "--json"]
            stdout, _ = self.run_cmd_in_container(cloc_cmd)
            cloc_data = json.loads(stdout)
            source_lines = cloc_data.get("SUM", {}).get("code", 0)

            print(
                f'[GENERATED .c FILE] | csmith_time: {csmith_time} | size: {source_size} | source_lines: {source_lines} ')

        except RuntimeError as e:
            print(f"Error during benchmark: {e}")
            # runt_ime, compile_time, exe_size, csmith_time, source_size, source_lines
            return 999999999, 999999999, 999999999, 999999999, 999999999, 999999999

        # run workload (compile and run .out file)
        compile_cmd = self.generate_compile_command(config)
        exe_file = self.output_file.replace(".c", ".out")
        try:
            # ==================compile stage====================
            # compile .out file
            start_compile = time.time()
            print(f"[RUN COMPILE CMD] | {compile_cmd}")
            stdout, stderr = self.run_cmd_in_container(compile_cmd, timeout=300)
            gcc_log = stdout + stderr
            end_compile = time.time()
            compile_time = end_compile - start_compile

            # get the size of compiled .out file (timeout for compiling is 300s)
            size_cmd = ["stat", "-c", "%s", exe_file]
            stdout, _ = self.run_cmd_in_container(size_cmd)
            exe_size = int(stdout.strip())
            print(f"[COMPILED .out FILE] | compiler_time: {compile_time} | exe_size: {exe_size}")

            # ==================execute stage====================
            # run the .out file (timeout for exe is 300s)
            start_run = time.time()
            run_cmd = ["./" + exe_file]
            print(f"[RUN EXECUTE CMD]: {run_cmd}")
            self.run_cmd_in_container(run_cmd, timeout=60)
            end_run = time.time()
            run_time = end_run - start_run
            print(f"[RUNTIME for EXECUTE .out FILE] | run_time: {run_time}")

            print(f"[PERFORMANCE]: run_time: {run_time}| compile_time: {compile_time}| exe_size: {exe_size}")
            return run_time, compile_time, exe_size, csmith_time, source_size, source_lines
        except RuntimeError as e:
            print(f"Error during compile/execute: {e}")
            return 999999999, 999999999, 999999999, 999999999, 999999999, 999999999
        except Exception as e:
            print(f"Error during compile/execute: {e}")
            return 999999999, 999999999, 999999999, 999999999, 999999999, 999999999
        finally:
            # ==================cleanup stage====================
            # clean up .c and .out file
            try:
                cleanup_cmd = ["rm", "-f", self.output_file, self.output_file.replace(".c", ".out")]
                print(f"[CLEANUP CMD] | {cleanup_cmd}")
                self.run_cmd_in_container(cleanup_cmd)
                print("[CLEANUP DONE] | Temporary files removed from container.")
                print("---------------------------------------------------------------")
            except Exception as e:
                print(f"[CLEANUP ERROR] | Failed to clean up files: {e}")

    def generate_csmith_command(self, factors):

        command_parts = [self.workload_bench, "--seed", str(self.seed), "-o", self.output_file]
        for key, value in factors.items():
            # Boolean flag: --flag (if true), skip if False
            if isinstance(value, bool):
                if value:
                    command_parts.append(key)
            # Value-based flags: --key value
            elif isinstance(value, (int, float, str)):
                # command_parts += [f"--{key} {value}"]
                command_parts += [f"--{key}", str(value)]

        return command_parts

    def generate_compile_command(self, config):
        exe_file = self.output_file.replace(".c", ".out")
        command_parts = [self.sys_name, "-I", self.include_file, self.output_file, "-o", exe_file]
        for key, value in config.items():
            if not self.is_flag_supported(f"-{key}"):  # check flag
                print(f"para is invalid: {key}")
                continue
            if isinstance(value, str):
                val_stripped = value.strip().lower()
                if val_stripped == "on":
                    command_parts.append(f"-{key}")
                elif val_stripped == "off":
                    continue  # skip
                else:
                    command_parts.append(f"-{key}={value}")
            else:
                # handle int, float, bool, etc.
                command_parts.append(f"-{key}={value}")

        return command_parts

    def run_cmd_in_container(self, cmd, timeout=60):
        """
        Run command inside the target docker container.

        :param timeout:
        :param cmd: Command list to execute inside the container (e.g., ['ls', '/app']).
        :return: stdout and stderr strings.
        """
        docker_cmd = ["docker", "exec", self.container_name] + cmd

        try:
            result = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=True
            )
            return result.stdout.decode('utf-8'), result.stderr.decode('utf-8')

        except CalledProcessError as e:
            raise RuntimeError(f"Command failed: {' '.join(docker_cmd)}\n"
                               f"Error: {e.stderr.decode('utf-8')}")


        except TimeoutExpired as e:
            self.target_system.restart_container()
            raise RuntimeError(f"[TIMEOUT]Command timed out after {timeout} seconds: {' '.join(docker_cmd)}")


    def is_flag_supported(self, flag: str) -> bool:
        check_cmd = [self.sys_name, flag, "-x", "c", "-c", "/dev/null", "-o", "/dev/null"]
        try:
            subprocess.run(
                ["docker", "exec", self.container_name] + check_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return True
        except CalledProcessError:
            return False
