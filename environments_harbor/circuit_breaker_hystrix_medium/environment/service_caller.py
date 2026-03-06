#!/usr/bin/env python3

import requests
import time

def fetch_data(url):
    """
    Fetch data from the given URL with infinite retries.
    This implementation will hang indefinitely if the service is down.
    """
    while True:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                try:
                    return response.json()
                except:
                    return response.text
            else:
                print(f"Request failed with status code: {response.status_code}")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(2)