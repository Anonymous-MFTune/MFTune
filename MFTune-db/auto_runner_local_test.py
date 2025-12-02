import multiprocessing
import subprocess
import time


def run_experiment(db_host, fidelity_type, tuning_method, system, run):
    """
    Dynamically pass parameters to each thread to run different algorithms
    :param db_host: selected db_host
    :param fidelity_type: selected fidelity type (single_/multi_fidelity)
    :param tuning_method:
    :return:
    """

    try:

        # Using Command Line Argument Passing Database Service and Tuning Methods
        result = subprocess.run(["python", "main.py",
                                 "--config", f"./params_setup/{system}_params_setup.ini",
                                 "--db_host", db_host,
                                 "--fidelity_type", fidelity_type,
                                 "--tuning_method", tuning_method,
                                 "--run", f"{run}"],
                                 check=True)
        print(f"ALGORITHM: {tuning_method}; TYPE: {fidelity_type}; SYS: {db_host}; completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during experiment with {tuning_method} on {db_host} ({fidelity_type}): {e}")
    except Exception as e:
        print(f"Unexpected error during experiment with {tuning_method} on {db_host} ({fidelity_type}): {e}")
    finally:
        # Stop and clean up services
        # docker_compose_down()
        print("-------------------")


def docker_compose_up(service_name):
    try:
        print("Starting docker services...")
        subprocess.run(["docker-compose", "up", "-d", f"{service_name}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting docker services: {e}")


def docker_compose_down():
    try:
        print("Stopping and cleaning up docker services...")
        subprocess.run(["docker-compose", "down", "--remove-orphans", "-v"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error cleaning docker services: {e}")


def stop_and_remove_all_containers():
    """
    Stop and remove all Docker containers (running or stopped).
    """
    try:
        print("Fetching all container IDs...")
        result = subprocess.run(["docker", "ps", "-aq"], stdout=subprocess.PIPE, text=True, check=True)
        container_ids = result.stdout.strip().split('\n')

        if container_ids and container_ids[0]:
            print("Stopping all containers...")
            subprocess.run(["docker", "stop"] + container_ids, check=True)

            print("Removing all containers...")
            subprocess.run(["docker", "rm"] + container_ids, check=True)

            print("All containers have been stopped and removed.")
        else:
            print("No containers to stop or remove.")
    except subprocess.CalledProcessError as e:
        print(f"Error while stopping/removing containers: {e}")



def stop_and_remove_container(container_name):
    """
    Stop and remove the specified container.
    """
    try:
        print(f"Stopping and removing container: {container_name}")
        subprocess.run(["docker", "stop", container_name], check=True)
        subprocess.run(["docker", "rm", container_name], check=True)
        print(f"Container {container_name} has been stopped and removed.")
    except subprocess.CalledProcessError as e:
        print(f"Error stopping/removing container {container_name}: {e}")


if __name__ == "__main__":

    run = 1
    # db service; fidelity_type; algorithm
    # experiments = [("mysql", "multi_fidelity", "data_analyser")]  # verify
    # experiments = [("postgres", "multi_fidelity", "data_analyser")]  # verify


    system = 'postgresql'
    # experiments = [("localhost", "multi_fidelity", "ga"),
    #                ("localhost", "single_fidelity", "ga"),
    #                ("localhost", "single_fidelity", "flash"),
    #                ("localhost", "single_fidelity", "bestconfig"),
    #                ("localhost", "single_fidelity", "smac"),
    #                ("localhost", "single_fidelity", "hyperband")
    #                ]
    experiments = [("localhost", "single_fidelity", "flash")]


    for i in range(run):
        for experiment in experiments:
            db_host, fidelity_type, tuning_method = experiment
            run_experiment(db_host, fidelity_type, tuning_method, system, i)

    print("All experiments completed.")
