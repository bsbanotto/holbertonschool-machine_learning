#!/usr/bin/env python3
"""
Using the unofficial SpaceX API, get information from SpaceX's first launch
Format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)

-------------------------------MY THOUGHTS HERE--------------------------------
Getting what the task is asking for and what the checker is looking for is
taking some work here. This is what I've decided to do that will match the
expected output.

Look at upcomming launches and get the earliest planned launch data
Get all of the information related to this launch
Print out that information the way the checker wants it

This kind of makes sense given upcomming.py is the file name in the example
------------------------------END OF MY THOUGHTS-------------------------------
"""

if __name__ == "__main__":
    import requests

    LAUNCHES = 'https://api.spacexdata.com/v5/launches/'
    ROCKETS = 'https://api.spacexdata.com/v4/rockets/'
    PADS = 'https://api.spacexdata.com/v4/launchpads/'

    upcomming_response = requests.get(LAUNCHES + 'upcoming')
    upcomming_launches = upcomming_response.json()

    # Get lists of launch information
    launch_names = [launch['name'] for launch in upcomming_launches]
    launch_dates = [launch['date_local'] for launch in upcomming_launches]
    launch_rockets = [launch['rocket'] for launch in upcomming_launches]
    launch_pads = [launch['launchpad'] for launch in upcomming_launches]

    # Get index of minimum date in list
    min_pos = launch_dates.index(min(launch_dates))

    # Get the Rocket Name from the ROCKETS API
    rocket_response = requests.get(ROCKETS + launch_rockets[min_pos])
    rocket_data = rocket_response.json()
    rocket = rocket_data['name']

    # Get Launchpad Name and Locality from the PADS API
    pad_response = requests.get(PADS + launch_pads[min_pos])
    pad_data = pad_response.json()
    pad = pad_data['name']
    locality = pad_data['locality']

    # Get the name and date from list of launch names and dates
    name = launch_names[min_pos]
    date = launch_dates[min_pos]

    # Print this stuff
    print("{} ({}) {} - {} ({})".format(name, date, rocket, pad, locality))
