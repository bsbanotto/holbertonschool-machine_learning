#!/usr/bin/env python3
"""
Using the unofficial SpaceX API, get information from SpaceX's first launc
Format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""

if __name__ == "__main__":
    import requests
    import datetime

    PAST_URL = 'https://api.spacexdata.com/v5/launches/past'
    
    past_response = requests.get(PAST_URL)

    past_launches = past_response.json()

    launch_ids = [launch['id'] for launch in past_launches]

    launch_names = [launch['name'] for launch in past_launches]
    launch_dates = [launch['date_utc'] for launch in past_launches]
    launch_rockets = [launch['rocket'] for launch in past_launches]
    launch_pads = [launch['launchpad'] for launch in past_launches]


    print("FalconSat (2006-03-25T10:30:00+12:00) Falcon 1 - Kwajalein Atoll (Omelek Island)")
