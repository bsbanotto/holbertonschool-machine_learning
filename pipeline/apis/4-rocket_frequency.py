#!/usr/bin/env python3
"""
This script will use the unofficial SpqceX API to get a count of launches per
rocket
"""
if __name__ == "__main__":
    import requests
    from collections import Counter

    URL = 'https://api.spacexdata.com/v3/launches'

    response = requests.get(URL)

    launch_data = response.json()

    rockets = [rocket['rocket']['rocket_name'] for rocket in launch_data]

    # Convert the list of rockets to a sorted dict
    launch_dict = dict(sorted(Counter(rockets).items(),
                              key=lambda x: x[1], reverse=True))

    for rocket, count in launch_dict.items():
        print(rocket + ': ' + str(count))
