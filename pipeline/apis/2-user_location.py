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

    # print(response.headers)

    user_data = response.json()

    if response.status_code == 200:
        print(user_data['location'])

    elif response.status_code == 404:
        print("Not found")

    elif response.status_code == 403:
        reset_time = response.headers['X-RateLimit-Reset']
        now = int(time.time())
        print("Reset in {} min".format((reset_time - now) // 60))
