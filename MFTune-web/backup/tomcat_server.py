import json
import os
import subprocess
import time
import shutil
import xml.etree.ElementTree as ET
import requests


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
            subprocess.run(f"echo {self.password} | sudo -S docker start {self.container_name}", shell=True, check=True)
            print(f"Container '{self.container_name}' started.")
        except Exception as e:
            print(f"Error starting Tomcat: {e}")
            return False

        # Waiting for tomcat starting (ping URL）
        for _ in range(10):
            try:
                response = requests.get(self.server_url, timeout=3)
                if response.status_code == 200:
                    print(f"Tomcat is up at {self.server_url}")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        print(f"Container started, but Tomcat is not responding at {self.server_url}")
        return False

    def stop_container(self):
        try:
            subprocess.run(f"echo {self.password} | sudo -S docker stop {self.container_name}", shell=True, check=True)
            print(f"{self.container_name} stopped")
        except Exception as e:
            print(f"Error stopping Tomcat: {e}")

    def set_server_knobs(self, knob_dict):
        try:
            # Copy original backup file to temporary file for modification
            subprocess.run(f"echo {self.password} | sudo -S cp {self.backup_path} {self.temp_path}", shell=True,
                           check=True)
            subprocess.run(f"echo {self.password} | sudo -S chmod 777 {self.temp_path}", shell=True, check=True)
        except Exception as e:
            print(f"Error preparing temp server.xml: {e}")
            return

        # Parse XML and set multiple attributes
        tree = ET.parse(f'{self.temp_path}')
        root = tree.getroot()
        connector = root.find('.//Connector')
        if connector is not None:
            print("Configuring config into temp.xml...")
            for attribute, new_value in knob_dict.items():
                if not isinstance(new_value, str):
                    new_value = str(new_value)
                connector.set(attribute, new_value)

        tree.write(f"{self.temp_path}", encoding='utf-8', xml_declaration=True)

        try:
            subprocess.run(f"echo {self.password} | sudo -S chmod 644 {self.temp_path}", shell=True, check=True)
            subprocess.run(
                f"echo {self.password} | sudo -S docker cp {self.temp_path} {self.container_name}:{self.config_file_path}",
                shell=True, check=True)
            print("Upload server.xml from local to server: SUCCESS")
        except Exception as e:
            print(f"Error copying updated config into container: {e}")

    def backup_config(self):
        try:
            print("Backup server.xml from server to local")
            subprocess.run(
                f"echo {self.password} | sudo -S docker cp {self.container_name}:{self.config_file_path} {self.backup_path}",
                shell=True, check=True)
        except Exception as e:
            print(f"Error modifying Tomcat: {e}")

    def restore_config(self):
        try:
            print("Upload server.xml from local to server ")
            subprocess.run(
                f"echo {self.password} | sudo -S docker cp {self.backup_path} {self.container_name}:{self.config_file_path}",
                shell=True, check=True)
        except Exception as e:
            print(f"Error modifying Tomcat: {e}")

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