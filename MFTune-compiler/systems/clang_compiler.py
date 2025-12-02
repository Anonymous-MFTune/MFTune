import json
import subprocess
import time
import shutil
import xml.etree.ElementTree as ET
import requests
import os
from subprocess import TimeoutExpired, CalledProcessError, Popen, PIPE
import shlex

class ClangCompiler:
    def __init__(self, args_compiler):
        self.container_name = args_compiler['container_name']
        self.knobs_info = self.initialize_knobs(args_compiler['knob_config_file'], int(args_compiler['knob_num']))
        self.use_sudo = None
        self.password = args_compiler.get('password')
        self.docker_image = args_compiler.get('dockerimage')

    def restart_container(self):
        try:
            subprocess.run(
                ["docker", "restart", self.container_name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"🔁 Container '{self.container_name}' restarted successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to restart container '{self.container_name}': {e.stderr.decode().strip()}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during restart: {str(e)}")
            return False

    @staticmethod
    def initialize_knobs(knob_config_file, knob_num):
        """initialize related information of knobs that need to be optimized"""
        global KNOBS
        global KNOB_INFO
        if knob_num == -1:
            f = open(knob_config_file)
            KNOB_INFO = json.load(f)
            KNOBS = list(KNOB_INFO.keys())
            f.close()
        else:
            f = open(knob_config_file)
            knob_tmp = json.load(f)
            i = 0
            KNOB_INFO = {}
            while i < knob_num:
                key = list(knob_tmp.keys())[i]
                KNOB_INFO[key] = knob_tmp[key]
                i = i + 1
            KNOBS = list(KNOB_INFO.keys())
            f.close()
        return KNOB_INFO

    def get_default_knobs(self):
        """obtain default knob settings according to self-define file"""
        default_knobs = {}
        for name, value in KNOB_INFO.items():
            if not value['type'] == "combination":
                default_knobs[name] = value['default']
            else:
                knobL = name.strip().split('|')
                valueL = value['default'].strip().split('|')
                for i in range(0, len(knobL)):
                    default_knobs[knobL[i]] = int(valueL[i])
        return default_knobs

    def start(self):
        """Start the Docker container (idempotent)."""
        print(f"INFO: Ensuring container '{self.container_name}' is running...")

        # check container status
        inspect_cmd = f"docker ps -a --filter name={self.container_name} --format '{{{{.Status}}}}'"
        result = self.execute_docker_command(inspect_cmd, check_output=True)
        status = result.stdout.strip().lower()

        if "up" in status:
            print(f"INFO: Container '{self.container_name}' is already running.")
            return True
        elif "exited" in status or "created" in status:
            self.execute_docker_command(f"docker start {self.container_name}")
            print(f"INFO: Container '{self.container_name}' restarted.")
            return True
        else:
            self.execute_docker_command(f"docker run -d --name {self.container_name} {self.docker_image}")
            print(f"INFO: Container '{self.container_name}' started new.")
            return True

    def execute_docker_command(self, command, check_output=True):
        """
        robust docker command executor: supports list or str (auto shlex split), auto handle sudo.
        """
        # 1) First-run docker permission probe (unchanged)
        if self.use_sudo is None:
            print("INFO: Checking Docker permissions for the first time...")
            try:
                subprocess.run(["docker", "ps"], check=True, capture_output=True, text=True, timeout=15)
                print("INFO: Docker can be run without sudo. Proceeding in non-sudo mode.")
                self.use_sudo = False
            except (CalledProcessError, TimeoutExpired):
                print("WARN: Docker direct access failed. Attempting with sudo...")
                try:
                    subprocess.run(["sudo", "docker", "ps"], check=True, capture_output=True, text=True, timeout=15)
                    print("INFO: Docker can be run with sudo.")
                    self.use_sudo = True
                except (CalledProcessError, TimeoutExpired) as sudo_e:
                    raise RuntimeError(f"FATAL: Cannot run docker, even with sudo. Error: {sudo_e}")
            except Exception as e:
                raise RuntimeError(f"FATAL: Unexpected error during Docker permission check: {e}")

        # 2) Normalize command into a list of argv
        if isinstance(command, str):
            command_parts = shlex.split(command)
        else:
            command_parts = command

        try:
            if self.use_sudo:
                if not self.password:
                    raise ValueError("Sudo is required, but no password was provided in the .ini file.")
                full_parts = ["sudo", "-S"] + command_parts
                process = Popen(full_parts, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True, encoding='utf-8')
                stdout_data, stderr_data = process.communicate(input=self.password + '\n')
                if check_output and process.returncode != 0:
                    raise CalledProcessError(process.returncode, full_parts, stdout_data, stderr_data)
                return subprocess.CompletedProcess(full_parts, process.returncode, stdout_data, stderr_data)
            else:
                return subprocess.run(command_parts, check=check_output, capture_output=True, text=True,
                                      encoding='utf-8')
        except CalledProcessError as e:
            print(f"ERROR: Docker command failed. Command: {' '.join(e.cmd)}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}",
                  file=os.sys.stderr)
            raise
        except Exception as e:
            raise RuntimeError(f"FATAL: Unexpected error during Docker command execution: {e}")