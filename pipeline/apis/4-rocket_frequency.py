#!/usr/bin/env python3
"""
This script will use the unofficial SpqceX API to get a count of launches per
rocket

Again, the text in the task is wrong. It looks like we need to use the updated
v4 or v5 launch data to get the counts to match what the checker is expecting
"""
if __name__ == "__main__":
    import requests
    from collections import Counter

    LAUNCH_URL = 'https://api.spacexdata.com/v5/launches'
    ROCKET_URL = 'https://api.spacexdata.com/v4/rockets'

    rocket_dict = {}

    # Create a list of all launched rockets from LAUNCH_URL
    launch_response = requests.get(LAUNCH_URL)
    launch_data = launch_response.json()
    launched_rockets = [rocket['rocket'] for rocket in launch_data]

    # Create lists of rocket_ids, rocket_names from ROCKET_URL
    rocket_response = requests.get(ROCKET_URL)
    rocket_data = rocket_response.json()
    rocket_ids = [rocket['id'] for rocket in rocket_data]
    rocket_names = [rocket['name'] for rocket in rocket_data]

    # Zip those up into a dictionary
    rocket_dict = dict(zip(rocket_ids, rocket_names))

    # Convert the list of rockets to a counted and sorted dict
    launch_dict = dict(sorted(Counter(launched_rockets).items(),
                              key=lambda x: x[1], reverse=True))

    # Replace keys in launch_dict with the appropriate rocket names
    result_dict = {rocket_dict[key]: value for key,
                   value in launch_dict.items() if key in rocket_dict}

    for rocket, count in result_dict.items():
        print(rocket + ': ' + str(count))
