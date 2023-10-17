#!/usr/bin/env python3
"""
Using the GitHub API, write a script that prints the location of a specific
user. If the status code is 403, print Reset in `X` minutes.
"""
if __name__ == "__main__":
    import requests
    import sys
    import time

    if len(sys.argv) != 2:
        print("Usage: ./2-user_location https://api.github.com/users/<user>")

    API = sys.argv[1]

    response = requests.get(API)

    user_data = response.json()

    if response.status_code == 200:
        if user_data['location']:
            print(user_data['location'])
        else:
            print("Not found")

    elif response.status_code == 403:
        rate_url = 'https://api.github.com/rate_limit'
        rate_response = requests.get(rate_url)
        rate_response_data = rate_response.json()
        reset_time = rate_response_data['rate']['reset']
        now = int(time.time())
        print("Reset in {} min".format((reset_time - now) // 60))
