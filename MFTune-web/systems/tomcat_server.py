import json
import os
import subprocess
import time
import shutil
import xml.etree.ElementTree as ET
import requests
from urllib.parse import urlparse
import requests, subprocess, socket, time

class TomcatServer:
    def __init__(self, args_server):
        self.container_name = args_server['container_name']
        self.config_file_path = args_server['config_file_path']
        self.backup_path = args_server['backup_path']
        self.password = args_server['password']
        self.server_url = args_server['url']
        self.temp_path = args_server['temp_config_file_path']
        self.knobs_info = self.initialize_knobs(args_server['knob_config_file'], int(args_server['knob_num']))

    def start_container(self):
        try:
            subprocess.run(["docker", "start", self.container_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Container '{self.container_name}' is started.")
        except Exception as e:
            print(f"Error starting container {self.container_name}: {e}")
            return False

        # Step 1: Waiting for DNS
        parsed = urlparse(self.server_url)
        hostname = parsed.hostname
        if not self.wait_dns_resolve(hostname):
            print(f"DNS resolution failed for {hostname}")
            return False

        # Step 2: Waiting for tomcat starting (ping URL）
        for _ in range(10):
            try:
                response = requests.get(self.server_url, timeout=3)
                if response.status_code == 200:
                    print(f"Tomcat startup successfully at {self.server_url}")
                    return True
            except requests.exceptions.RequestException:
                time.sleep(2)

        print(f"Container started, but Tomcat is not responding at {self.server_url}")
        return False

    def stop_container(self):
        try:
            subprocess.run(["docker", "stop", self.container_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"{self.container_name} is stopped")
            return True
        except Exception as e:
            print(f"Error stopping container {self.container_name}: {e}")
            return False
    def wait_dns_resolve(self, host, retries=10, delay=2):
        for _ in range(retries):
            try:
                ip = socket.gethostbyname(host)
                print(f"[DNS Ready] {host} resolved to {ip}")
                return True
            except socket.gaierror:
                print(f"[DNS Not Ready] Retrying...")
                time.sleep(delay)
        return False

    def set_server_knobs(self, knob_dict):
        # TODO: add a apachectl configtest, ensuring the config is validate
        try:
            # Copy default file to temporary file for modification
            shutil.copyfile(self.backup_path, self.temp_path)
            os.chmod(self.temp_path, 0o777)
            print("[1] default.xml [tuning container] -> temp.xml [tuning container]")
        except Exception as e:
            print(f"Error copying default.xml to temp.xml: {e}")
            return False

        # Parse XML and set multiple attributes
        tree = ET.parse(f'{self.temp_path}')
        root = tree.getroot()
        connector = root.find('.//Connector')
        if connector is not None:
            print("[2] Writing config into temp.xml [tuning container]")
            for attribute, new_value in knob_dict.items():
                if not isinstance(new_value, str):
                    new_value = str(new_value)
                connector.set(attribute, new_value)

        tree.write(f"{self.temp_path}", encoding='utf-8', xml_declaration=True)

        try:
            os.chmod(self.temp_path, 0o644)
            subprocess.run(["docker", "cp", self.temp_path, f"{self.container_name}:{self.config_file_path}"],
                           check=True)
            print("[3] PUSH: temp.xml [tuning container] -> server.xml [system container]: SUCCESS")
            return True
        except Exception as e:
            print(f"Error copying updated config into system container: {e}")
            return False

    def backup_config(self):
        try:
            print("PULL: server.xml [system container] -> default.xml [tuning container]")
            subprocess.run([
                "docker", "cp",
                f"{self.container_name}:{self.config_file_path}",
                self.backup_path
            ], check=True)
            return True
        except Exception as e:
            print(f"Error pulling .xml: {e}")
            return False

    def restore_config(self):
        try:
            print("PUSH: temp.xml [tuning container] -> server.xml [system container]")
            subprocess.run([
                "docker", "cp",
                self.backup_path,
                f"{self.container_name}:{self.config_file_path}"
            ], check=True)
            return True
        except Exception as e:
            print(f"Error pushing .xml: {e}")
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