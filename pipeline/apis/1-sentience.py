#!/usr/bin/env python3
"""
Module to get a list of all planets that are home to sentient species
"""
import requests


API_ROOT = "https://swapi-api.alx-tools.com/api/"


def sentientPlanets():
    """
    Returns:
        list of planets that are home to sentient species
    """
    page = 1
    species_planet_urls = []
    planet_list = []

    response = requests.get(API_ROOT + 'species/?page={}'.format(page))
    species_data = response.json()

    while species_data['next']:
        response = requests.get(API_ROOT + 'species/?page={}'.format(page))
        species_data = response.json()
        for species in species_data['results']:
            if species['designation'] == 'sentient':
                species_planet_urls.extend([species['homeworld']])
            if species['classification'] == 'sentient':
                species_planet_urls.extend([species['homeworld']])
        # species_planet_urls.extend([species['homeworld'] for species in species_data['results']])
        page += 1

    # print(len(species_planet_urls))

    for url in species_planet_urls:
        if url is not None:
            response = requests.get(url)
            planet_data = response.json()
            planet_list.append(planet_data['name'])

    return(planet_list)


if __name__ == "__main__":
    planets = sentientPlanets()
    for planet in planets:
        print(planet)
    # sentientPlanets()