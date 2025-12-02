import json
import time
import psycopg2
import subprocess
from utils.db_connector import PostgresqlConnector
from psycopg2 import sql


class PostgresqlDB:
    def __init__(self, args):
        self.args = args

        # PostgresSQL Info
        self.host = args['host']
        self.port = args['port']
        self.user = args['user']
        self.passwd = args['password']
        self.dbname = args['dbname']
        self.container_name = f"{args['host']}_container"
        self.db_connector = PostgresqlConnector(self.host, self.port, self.user, self.passwd, self.dbname)

        # Postgresql Knobs
        self.knobs_info = self.initialize_knobs(args['knob_config_file'], int(args['knob_num']))
        self.default_knobs = self.get_default_knobs()


    def restart_container(self):
        """Restart the Docker container running Postgresql."""
        try:
            print(f"Restarting container: {self.container_name}")
            # Stop the container
            subprocess.run(["docker", "restart", self.container_name], check=True)
            print(f"Container '{self.container_name}' restarted successfully.")

            if self.wait_until_postgresql_ready(timeout=60):
                # Close old connector
                if self.db_connector:
                    self.db_connector.close_db()

                # Reinitialize connector
                self.db_connector = PostgresqlConnector(self.host, self.port, self.user, self.passwd, self.dbname)
                print("[INFO] MySQL connector re-initialized after restart.")
            else:
                print("[ERROR] PostgreSQL restart failed or too slow to respond.")
        except subprocess.CalledProcessError as e:
            print(f"Error restarting container '{self.container_name}': {e}")

    def initialize_knobs(self, knob_config_file, knob_num):
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

    def get_current_db_configurations(self):
        """Get the current configurations of PostgreSQL"""
        sql = "SHOW ALL;"
        configurations = self.db_connector.execute(sql)
        return {config[0]: config[1] for config in configurations}

    def set_db_knob(self, config):

        conn_count = self.check_active_connections()
        if conn_count != -1:
            print(f"[DEBUG] Current Postgresql connection count: {conn_count}")

        """Set the configs for PostgreSQL with connection check and container restart fallback."""
        if not self.check_connection_alive():
            print(f"[WARNING] Connection to {self.container_name} failed. Restarting container...")
            self.restart_container()
            time.sleep(5)  # wait for PostgreSQL to be ready
        try:
            """Set a configuration knob in PostgreSQL and verify the change."""
            self.db_connector.connect_db()
            # Enable autocommit mode for ALTER SYSTEM
            self.db_connector.conn.autocommit = True
            for knob_name, knob_value in config.items():
                sql = f"ALTER SYSTEM SET {knob_name} = '{knob_value}';"
                self.db_connector.execute(sql)
                # Reload configuration to apply changes
                self.db_connector.execute("SELECT pg_reload_conf();")
                print(f"[Knob Setting] Set {knob_name} = {knob_value}")

            self.db_connector.close_db()
        except Exception as e:
            print(f"[ERROR] Failed to set knobs: {e}")

    def manage_database(self, action, dbname="testdb"):
        """Execute database management actions such as DROP DATABASE and CREATE DATABASE."""
        # [IMPORTANT]: Connect to a different database, like `postgres`; so that we can drop test db
        self.db_connector = PostgresqlConnector(self.host, self.port, self.user, self.passwd, "postgres")
        conn = self.db_connector.connect_db()
        conn.autocommit = True  # Enable autocommit to avoid transaction block errors

        try:
            cur = conn.cursor()
            if action == "drop":
                # Terminate all connections to `dbname`
                terminate_sql = f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{dbname}';"
                cur.execute(terminate_sql)
                # Drop the database if it exists
                cur.execute(f"DROP DATABASE IF EXISTS {dbname};")
                print(f"Database '{dbname}' dropped successfully.")

            elif action == "create":
                # Create the database if it does not exist
                cur.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(self.dbname)
                ))
                print(f"Database '{dbname}' created successfully.")

            elif action == "uuid":
                with psycopg2.connect(host=self.host, port=self.port, user=self.user, password=self.passwd, database=dbname) as conn:
                    conn.autocommit = True
                    with conn.cursor() as cursor:
                        cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
                        cursor.execute("""
                            DO $$ DECLARE r RECORD;
                            BEGIN
                                FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'sbtest%') LOOP
                                    EXECUTE format('
                                        ALTER TABLE %I ADD COLUMN uuid_id UUID DEFAULT uuid_generate_v4();
                                        UPDATE %I SET uuid_id = uuid_generate_v4();
                                        ALTER TABLE %I DROP CONSTRAINT IF EXISTS %I_pkey;
                                        ALTER TABLE %I ADD PRIMARY KEY (uuid_id);
                                        ALTER TABLE %I DROP COLUMN id;
                                        ALTER TABLE %I RENAME COLUMN uuid_id TO id;
                                    ', r.tablename, r.tablename, r.tablename, r.tablename, r.tablename, r.tablename, r.tablename);
                                END LOOP;
                            END $$;
                        """)
                        print(f"All sbtest tables in {dbname} now use UUID as primary keys.")
            cur.close()
        except Exception as e:
            print(f"Error executing action '{action} {dbname}': {e}")
        finally:
            self.db_connector.close_db()




    def get_db_knob_value(self, knob_name):
        """Retrieve the current value of a specific PostgreSQL knob"""
        self.db_connector.connect_db()
        sql = f"SHOW {knob_name};"
        result = self.db_connector.execute(sql)
        # Debugging: Print the result to verify structure
        # print(f"Result of SHOW {knob_name}: {result}")
        # Access the first element directly if result is valid
        self.db_connector.close_db()
        return result[0][0] if result else None

    def check_active_connections(self, print_details=False):
        """
        Check current number of user connections to PostgreSQL, excluding internal background processes.
        """
        try:
            conn = self.db_connector.connect_db()
            cur = conn.cursor()
            cur.execute("""
                SELECT pid, usename, datname, state, client_addr, backend_type
                FROM pg_stat_activity
                WHERE backend_type = 'client backend'
                  AND datname = %s;
            """, (self.dbname,))
            rows = cur.fetchall()
            self.db_connector.close_db()

            conn_count = len(rows)
            if print_details:
                print(f"[INFO] Active user connections to '{self.dbname}': {conn_count}")
                for row in rows:
                    print(row)

            return conn_count

        except Exception as e:
            print(f"[ERROR] Failed to check PostgreSQL connections: {e}")
            return -1

    def wait_until_postgresql_ready(self, timeout=60):
        """Wait until PostgreSQL is ready to accept connections."""
        start = time.time()
        while time.time() - start < timeout:
            if self.check_connection_alive():
                print("[INFO] PostgreSQL is ready to accept connections.")
                return True
            time.sleep(5)
        print("[ERROR] PostgreSQL did not become ready in time.")
        return False

    def check_connection_alive(self):
        try:
            temp_connector = PostgresqlConnector(self.host, self.port, self.user, self.passwd, self.dbname)
            conn = temp_connector.connect_db()
            if conn:
                temp_connector.close_db()
                return True
        except Exception:
            return False
        return False
