import os
import subprocess
import time
import shutil
import sys


def docker_db_stop(db_container):
    try:
        subprocess.run(["docker", "stop", db_container])
    except subprocess.CalledProcessError as e:
        print(f"Error during stop {db_container}: {e}")


def setup_logging():
    run = os.environ.get("RUN", "0")
    tuning_method = os.environ.get("TUNING_METHOD", "unknown")
    compiler_host = os.environ.get("COMPILER_HOST", "unknown")

    log_dir = "/app/logs"
    log_file = f"{log_dir}/run{run}_{tuning_method}_{compiler_host}_console.log"

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
        print(f"LOG STARTED | RUN={run} | METHOD={tuning_method} | HOST={compiler_host}", flush=True)
        print("=" * 80, flush=True)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Logging initialized.\n", flush=True)

    except Exception as e:
        print(f"Failed to setup logging: {e}")


def run_experiment():

    compiler_host = os.environ["COMPILER_HOST"]
    tuning_method = os.environ["TUNING_METHOD"]
    fidelity_type = os.environ["FIDELITY_TYPE"]
    run = os.environ["RUN"]
    system = os.environ.get("SYSTEM", "gcc")


    print(f"Running tuning: {tuning_method} | Compiler: {compiler_host} | Fidelity: {fidelity_type} | Run: {run}")
    time.sleep(20)  # Given enough time for starting SQL service
    try:
        subprocess.run(["python3", "main.py",
                                 "--config", f"./params_setup/{system}_params_setup.ini",
                                 "--fidelity_type", fidelity_type,
                                 "--tuning_method", tuning_method,
                                 "--run", f"{run}",
                                 "--service_name", f"{compiler_host}",
                                 "--container_name", f"{compiler_host}_container"],
                                check=True,
                                stdout=sys.stdout,  # redirect output
                                stderr=sys.stderr  # redirect error
                       )
        print(f"Tuning method '{tuning_method}' finished for run {run}")
    except subprocess.CalledProcessError as e:
        print(f"Error during tuning execution: {e}")
    finally:
        docker_db_stop(f"{compiler_host}_container")


if __name__ == "__main__":
    setup_logging()
    run_experiment()
