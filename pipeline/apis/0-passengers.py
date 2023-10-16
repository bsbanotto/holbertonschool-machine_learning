#!/usr/bin/env python3
"""
Module to get starship data from SWAPI. Will return only ships capable of
carrying more than the given number of passengers, or empty list if no ships
available
"""
import requests
import json


API_ROOT = "https://swapi-api.alx-tools.com/api/"


def availableShips(passengerCount):
    """
    Args:
        passengerCount (int): minimum number of passengers a ship should hold

    Returns:
        List of ships
    """
    page = 1
    results = []
    ships_list = []
    passengers_list = []
    response = requests.get(API_ROOT + 'starships/?page={}'.format(page))
    ships_data = response.json()

    while ships_data['next']:
        response = requests.get(API_ROOT + 'starships/?page={}'.format(page))
        ships_data = response.json()
        ships_list.extend([ship['name'] for ship in ships_data['results']])
        passengers_list.extend([ship['passengers']
                                for ship in ships_data['results']])
        page += 1

    for i in range(0, len(passengers_list)):
        passengers_list[i] = passengers_list[i].replace(',', '')
        if passengers_list[i] == 'n/a':
            passengers_list[i] = -1
        if passengers_list[i] == 'unknown':
            passengers_list[i] = -1
        passengers_list[i] = int(passengers_list[i])

    ship_passenger_dict = dict(zip(ships_list, passengers_list))

    for ship, passengers in ship_passenger_dict.items():
        if int(passengers) >= passengerCount:
            results.append(ship)

    return results


if __name__ == "__main__":
    ships = availableShips(4)
    for ship in ships:
        print(ship)
