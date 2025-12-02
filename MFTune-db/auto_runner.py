import os
import subprocess
import sys
import time
import shutil


def docker_db_stop(db_container):
    try:
        subprocess.run(["docker", "stop", db_container])
    except subprocess.CalledProcessError as e:
        print(f"Error during stop {db_container}: {e}")


def copy_lua_scripts(system):
    """
    Copy customized .lua scripts for the target system to /usr/share/sysbench/
    """
    src_dir = os.path.join("lua", system.lower())  # e.g., lua/mysql or lua/postgresql
    dst_dir = "/usr/share/sysbench"

    if os.path.exists(src_dir):
        for filename in os.listdir(src_dir):
            if filename.endswith(".lua"):
                src_file = os.path.join(src_dir, filename)
                dst_file = os.path.join(dst_dir, filename)
                try:
                    shutil.copy(src_file, dst_file)
                    print(f"[Lua Setup] Copied {src_file} -> {dst_file}")
                except Exception as e:
                    print(f"[Lua Setup] Failed to copy {src_file}: {e}")
    else:
        print(f"[Lua Setup] Skipping .lua copy — folder '{src_dir}' not found.")


def setup_logging():
    run = os.environ.get("RUN", "0")
    tuning_method = os.environ.get("TUNING_METHOD", "unknown")
    db_host = os.environ.get("DB_HOST", "server")

    log_dir = "/app/logs"
    log_file = f"{log_dir}/run{run}_{tuning_method}_{db_host}_console.log"

    try:

        os.makedirs(log_dir, exist_ok=True)
        # check rights for write/read
        if not os.access(log_dir, os.W_OK):
            print(f"No write permission to {log_dir}, attempting chmod...")
            try:
                os.chmod(log_dir, 0o777)
            except Exception as chmod_err:
                print(f"chmod failed: {chmod_err}")

        if not os.access(log_dir, os.W_OK):
            print(f"Still no write permission to {log_dir}. Logging disabled.")
            return

        # Open log file and redirect (flush)
        log_fh = open(log_file, "a", buffering=1)
        sys.stdout = log_fh
        sys.stderr = log_fh

        print("=" * 80, flush=True)
        print(f"LOG STARTED | RUN={run} | METHOD={tuning_method} | HOST={db_host}", flush=True)
        print("=" * 80, flush=True)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Logging initialized.\n", flush=True)

    except Exception as e:
        print(f"Failed to setup logging: {e}")


def run_experiment():

    db_host = os.environ["DB_HOST"]
    fidelity_type = os.environ["FIDELITY_TYPE"]
    tuning_method = os.environ["TUNING_METHOD"]
    system = os.environ.get("SYSTEM", "mysql")
    run = os.environ.get("RUN", "0")
    copy_lua_scripts(system)

    print(f"Running tuning: {tuning_method} | DB: {db_host} | Fidelity: {fidelity_type} | Run: {run}")
    time.sleep(20)  # Given enough time for starting SQL service
    try:
        subprocess.run([
            "python", "main.py",
            "--config", f"./params_setup/{system}_params_setup.ini",
            "--db_host", db_host,
            "--fidelity_type", fidelity_type,
            "--tuning_method", tuning_method,
            "--run", str(run)
        ], check=True,
           stdout=sys.stdout,
           stderr=sys.stderr
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during tuning execution: {e}")
    finally:
        docker_db_stop(f"{db_host}_container")


if __name__ == "__main__":
    setup_logging()
    run_experiment()

