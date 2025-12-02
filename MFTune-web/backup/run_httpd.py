import multiprocessing
import subprocess
import time
from utils.docker_utils import DockerUtils



def run_experiment(password, service_name, server_port, fidelity_type, tuning_method, system, run):
    """
    Dynamically pass parameters to each thread to run different algorithms
    :param service_name:
    :param server_port:
    :param fidelity_type:
    :param tuning_method:
    :param system:
    :param run:
    :return:
    """

    try:
        DockerUtils.docker_compose_up(service_name, password)
        time.sleep(10)
        # Using Command Line Argument Passing Database Service and Tuning Methods
        result = subprocess.run(["python3", "main.py",
                                 "--config", f"./params_setup/{system}_params_setup.ini",
                                 "--server_port", server_port,
                                 "--fidelity_type", fidelity_type,
                                 "--tuning_method", tuning_method,
                                 "--run", f"{run}",
                                 "--container_name", f"{service_name}_container"],
                                check=True)
        print(f"RUN {run} | METHOD: {tuning_method}| TYPE: {fidelity_type} | COMPLETED SUCCESSFULLY.")
    except subprocess.CalledProcessError as e:
        print(f"Error during experiment with {tuning_method} on {system} ({fidelity_type}): {e}")
    except Exception as e:
        print(f"Unexpected error during experiment with {tuning_method} on {system} ({fidelity_type}): {e}")
    finally:
        DockerUtils.docker_compose_down()


if __name__ == "__main__":

    run = 4
    system = 'httpd'
    # Host password for privilege control
    password = 'mosheng126'
    # service name; port; type; tuning method
    experiments = [("httpd_bestconfig", "9081", "single_fidelity", "bestconfig"),
                   ("httpd_smac", "9082", "single_fidelity", "smac"),
                   ("httpd_ga_mf", "9083", "multi_fidelity", "ga"),
                   ("httpd_ga_sf", "9084", "single_fidelity", "ga"),
                   ("httpd_flash", "9085", "single_fidelity", "flash"),
                   ("httpd_hyperband", "9086", "multi_fidelity", "hyperband"),
                   ]

    for i in range(run):
        for experiment in experiments:
            service_name, server_port, fidelity_type, tuning_method = experiment
            print(f"=========== RUN {i} START ===========")
            print(f"RUN {i} | METHOD: {tuning_method} | FIDELITY: {fidelity_type} | SYSTEM: {system} | SERVER_SERVICE: {service_name} | PORT: {server_port}")
            run_experiment(password, service_name, server_port, fidelity_type, tuning_method, system, i)

    print("ALL EXPERIMENTS ARE COMPLETED.")
