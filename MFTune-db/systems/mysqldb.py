import subprocess
import time
import shlex
import os
import json
from utils.db_connector import DBConnector
from utils.db_connector import MysqlConnector
import re


class MysqlDB:
    def __init__(self, args):
        self.args = args

        # MySQL Info
        self.host = args['host']
        self.port = args['port']
        self.user = args['user']
        self.password = args['password']
        self.dbname = args['dbname']
        self.ssl_pro = args['ssl_pro']
        self.container_name = f"{args['host']}_container"
        self.db_connector = MysqlConnector(self.host, self.port, self.user, self.password, self.dbname, self.ssl_pro)

        # MySQL Knobs
        self.knobs_info = self.initialize_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = self.get_default_knobs()

    def restart_container(self):
        """Restart the Docker container running MySQL."""
        try:
            print(f"Restarting container: {self.container_name}")
            subprocess.run(["docker", "restart", self.container_name], check=True)
            print(f"Container '{self.container_name}' restarted successfully.")

            if self.wait_until_mysql_ready(timeout=60):
                # Close old connector
                if self.db_connector:
                    self.db_connector.close_db()

                # Reinitialize connector
                self.db_connector = MysqlConnector(
                    self.host, self.port, self.user, self.password, self.dbname, self.ssl_pro
                )
                print("[INFO] MySQL connector re-initialized after restart.")
            else:
                print("[ERROR] MySQL restart failed or too slow to respond.")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Restarting container '{self.container_name}' failed: {e}")

    def initialize_knobs(self, knob_config_file, knob_num):
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


    # TODO: need to be specified
    def get_current_db_configurations(self):
        """get the current configs of mysql"""
        sql = "SHOW VARIABLES WHERE Variable_name IN （'innodb_buffer_pool_size', 'innodb_log_file_size', ...);"  # ...indicates other variable
        configurations = self.db_connector.execute(sql)
        return {config[0]: config[1] for config in configurations}

    def set_db_knob(self, config):
        """set the configs for mysql"""
        conn_count = self.check_active_connections()
        if conn_count != -1:
            print(f"[DEBUG] Current MySQL connection count: {conn_count}")

        """Set the configs for MySQL with connection check and container restart fallback."""
        if not self.check_connection_alive():
            print(f"[WARNING] Connection to {self.container_name} failed. Restarting container...")
            self.restart_container()
            time.sleep(5)  # wait for MySQL to be ready
        try:
            # getting connection before setting the value of knob; after that, close connection
            conn = self.db_connector.connect_db()
            for knob_name, knob_value in config.items():
                sql = f"SET GLOBAL {knob_name} = {knob_value};"
                self.db_connector.execute(sql)
                print(f"[Knob Setting] Set {knob_name} = {knob_value}")
            self.db_connector.close_db()
        except Exception as e:
            print(f"[ERROR] Failed to set knobs: {e}")

    def manage_database(self, action, dbname="testdb"):

        """Manage MySQL database within Docker, with robust cleanup of both logical and physical remnants."""

        conn = self.db_connector.connect_db()
        try:
            cur = conn.cursor()

            if action == "drop":
                try:
                    cur.execute(f"DROP DATABASE IF EXISTS {dbname};")
                    print(f"[INFO] SQL DROP DATABASE '{dbname}' executed successfully.")
                except Exception as sql_drop_error:
                    print(f"[WARNING] SQL DROP DATABASE '{dbname}' failed: {sql_drop_error}")

                # check if dbname exists, if so, try to delete it
                check_cmd = f"docker exec {self.container_name} bash -c 'test -d /var/lib/mysql/{dbname}'"
                if subprocess.run(check_cmd, shell=True).returncode == 0:
                    print(f"[INFO] Cleaning up Docker filesystem for '{dbname}'...")
                    docker_cmd = f"docker exec {self.container_name} bash -c 'rm -rf /var/lib/mysql/{dbname}'"
                    result = subprocess.run(docker_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if result.returncode != 0:
                        print(f"[WARNING] Failed to delete physical files: {result.stderr.decode().strip()}")
                    else:
                        print(f"[INFO] Removed /var/lib/mysql/{dbname} from Docker.")
                else:
                    print(f"[INFO] No physical files found for '{dbname}'. Skipping cleanup.")

            if action == "create":
                try:
                    cur.execute(f"CREATE DATABASE IF NOT EXISTS {dbname};")
                    print(f"[INFO] Created database '{dbname}'.")
                except Exception as create_error:
                    print(f"[ERROR] Failed to create database '{dbname}': {create_error}")

            if action == "restart":
                self.restart_mysql(conn)

            cur.close()

        except Exception as e:
            print(f"[ERROR] manage_database('{action}', '{dbname}') failed: {e}")
        finally:
            self.db_connector.close_db()

    def restart_mysql(self, conn):
        """Restart MySQL server using SQL commands."""
        try:
            cur = conn.cursor()
            # Issue the SHUTDOWN command
            cur.execute("SHUTDOWN;")
            print("MySQL server shutdown successfully.")
            time.sleep(5)

            # Reconnect to the MySQL server
            self.db_connector = MysqlConnector(self.host, self.port, self.user, self.password, "mysql", self.ssl_pro)
            conn = self.db_connector.connect_db()
            print("MySQL server restarted successfully.")
        except Exception as e:
            print(f"Error restarting MySQL server: {e}")

    def get_db_knob_value(self, knob_name):

        conn = self.db_connector.connect_db()
        cur = conn.cursor()
        sql = "show variables like '{}';".format(knob_name)
        cur.execute(sql)
        result = cur.fetchall()
        self.db_connector.close_db()
        return result[0][1]

    def check_active_connections(self, print_details=False):
        """
        Check current number of active connections to MySQL.
        Returns: conn_count (int): number of active connections
        """
        try:
            conn = self.db_connector.connect_db()
            cur = conn.cursor()
            cur.execute("SHOW PROCESSLIST;")
            rows = cur.fetchall()
            self.db_connector.close_db()

            conn_count = len(rows)
            if print_details:
                print(f"[INFO] Active MySQL connections: {conn_count}")
                for row in rows:
                    print(row)

            return conn_count

        except Exception as e:
            print(f"[ERROR] Failed to check MySQL connections: {e}")
            return -1

    def wait_until_mysql_ready(self, timeout=60):
        """Wait until MySQL is ready to accept connections."""
        start = time.time()
        while time.time() - start < timeout:
            if self.check_connection_alive():
                print("[INFO] MySQL is ready to accept connections.")
                return True
            time.sleep(5)
        print("[ERROR] MySQL did not become ready in time.")
        return False

    def check_connection_alive(self):
        try:
            temp_connector = MysqlConnector(
                self.host, self.port, self.user, self.password, self.dbname, self.ssl_pro
            )
            conn = temp_connector.connect_db()
            if conn:
                temp_connector.close_db()
                return True
        except Exception:
            return False
        return False
