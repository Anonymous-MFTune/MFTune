import subprocess


class DockerUtils:

    @staticmethod
    def docker_compose_up(service_name):
        """
        Bring up a specific service using docker-compose in detached mode.
        """
        try:
            print(f"[Docker] Starting service: {service_name}")
            subprocess.run(["docker-compose", "up", "-d", service_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Docker Error] Failed to start service '{service_name}': {e}")

    @staticmethod
    def docker_compose_down():
        """
        Bring down all services and remove volumes.
        """
        try:
            print("[Docker] Shutting down services and removing volumes...")
            subprocess.run(["docker-compose", "down", "--remove-orphans", "-v"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Docker Error] Failed to shut down services: {e}")
