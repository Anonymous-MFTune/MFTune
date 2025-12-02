import requests
import time


class ServerConnector:
    def __init__(self, password, server_url):

        self.password = password
        self.server_url = server_url

    def connect_with_retry(self, retries=5, delay=5):
            """
            Attempts to connect to the server URL with retries.
            
            :param retries: Number of retry attempts.
            :param delay: Delay (in seconds) between retries.
            :return: True if the server is responsive, raises Exception otherwise.
            """
            for i in range(retries):
                try:
                    response = requests.get(self.server_url, timeout=10)
                    if response.status_code == 200:
                        print(f"Connect to {self.server_url}: SUCCESS")
                        return True
                    else:
                        print(f"Unexpected status code: {response.status_code}")
                except requests.RequestException as e:
                    print(f"Error: {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
            print(f"Failed to connect to {self.server_url} after {retries} retries.")
            return False

    def connect_and_validate(self):
        """
        Connects to the server URL and validates the response.
        
        :return: True if the server responds with a status code 200.
        :raises: Exception if the server is unresponsive or responds with an unexpected status code.
        """
        try:
            response = requests.get(self.server_url, timeout=10)
            if response.status_code == 200:
                print(f"Server {self.server_url} is responsive.")
                return True
            else:
                print(f"Server responded with unexpected status code: {response.status_code}")
                raise Exception(f"Unexpected response from server: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error: Unable to connect to {self.server_url}.")
            raise Exception(f"Failed to connect to server: {e}")