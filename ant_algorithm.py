import random
import pandas as pd
import numpy as np
import matplotlib as plt
import pulp
import itertools
import gmaps
import googlemaps
import haversine as hs


def load_coordinates(file_path):
    df = pd.read_excel(file_path)
    cities = df['City'].tolist()
    demands = df['Demand'].tolist()
    latitudes = df['Latitude'].tolist()
    longitudes = df['Longitude'].tolist()

    return cities, demands, latitudes, longitudes


results1 = load_coordinates('Locations.xlsx')
cities1 = results1[0]
demands1 = results1[1]
latitudes1 = results1[2]
longitudes1 = results1[3]


def transform_data(cities, demands, latitudes, longitudes):
    transformed = {}
    for i in range(len(cities)):
        transformed[cities[i]] = [demands[i], (latitudes[i], longitudes[i])]

    return transformed


transformed1 = transform_data(cities1, demands1, latitudes1, longitudes1)


def create_distance_matrix(latitudes, longitudes):
    if len(latitudes) != len(longitudes):
        return 'Number of input latitudes must be equal to number of input longitudes'

    distance_matrix = []
    for i in range(len(latitudes)):
        row = []
        for j in range(len(latitudes)):
            loc1 = (latitudes[i], longitudes[i])
            loc2 = (latitudes[j], longitudes[j])
            distance = hs.haversine(loc1, loc2)
            row.append(distance)
        distance_matrix.append(row)

    return distance_matrix


def create_weights_matrix(n_latitudes, n_longitudes):
    ones = np.ones((n_latitudes, n_longitudes))
    normalized = ones / (n_latitudes * n_longitudes)
    return normalized


def create_ants(n_ants, capacities, start_index):
    if n_ants != len(capacities):
        print('Nr of ants doesn\'t much length of capacities')
    ants = []
    for i in range(n_ants):
        ants.append([i, capacities[i], start_index])

    return ants


'''vehicles1 = create_vehicles(5, [10, 10, 10, 10, 10])
for v in vehicles1:
    print(v)'''


def select_destination(distances, weights, delivered, capacity, demands):
    index, distance = np.nan, np.nan
    for i in range(len(weights)):
        distance = random.choices(distances, weights)[0]
        index = distances.index(distance)
        if index not in delivered and demands[index] <= capacity:
            break

    return index, distance


def deliver(latitudes, longitudes, demands, n_ants, capacities, start_index, n_groups=1):
    d_matrix = create_distance_matrix(latitudes, longitudes)
    d_weights = create_weights_matrix(len(latitudes), len(longitudes))
    original_capacities = capacities

    best_group = [float('inf'), []]

    for group in range(n_groups):
        # make a group run
        delivered = [start_index]
        ants = create_ants(n_ants, capacities, start_index)
        ants_routes = {}
        for ant in ants:
            ants_routes[ant[0]] = {'Locations': [start_index], 'Distance': 0}

        while len(delivered) != len(demands):
            for ant in ants:
                des_id, distance = select_destination(d_matrix[ant[2]], d_weights[ant[2]], delivered, ant[1], demands)
                delivered.append(des_id)
                ant[2] = des_id
                ants_routes[ant[0]]['Locations'].append(des_id)
                ants_routes[ant[0]]['Distance'] += distance

        for ant in ants:
            d_to_startpoint = d_matrix[ant[2]][0]
            ants_routes[ant[0]]['Locations'].append(start_index)
            ants_routes[ant[0]]['Distance'] += d_to_startpoint

        # store group run results
        group_total_distance = 0
        for ant in ants_routes.keys():
            group_total_distance += ants_routes[ant]['Distance']
        print(f'Group {group}: {group_total_distance}')

        if group_total_distance < best_group[0]:
            best_group[0] = group_total_distance
            best_group[1] = []
            for ant in ants_routes.keys():
                best_group[1].append(ants_routes[ant]['Locations'])

    return best_group


def pprint(dictionary):
    total_distance = 0
    for key, value in dictionary.items():
        print(f'{key}: {value}\n')
        total_distance += value['Distance']
    print(f'Total distance is {round(total_distance)} km.')


def ant_algorithm(latitudes, longitudes, demands, start_index, n_ants, capacities, n_groups, alpha, n):
    matrix = create_distance_matrix(latitudes, longitudes)
    original_capacities = capacities
    p = np.ones(latitudes * longitudes).reshape(latitudes, longitudes) / (latitudes * longitudes)

    for i in range(n):
        for group in n_groups:
            delivered = [0]
            for ant in n_ants:
                if len(delivered) == len(latitudes):
                    break


results = deliver(latitudes1, longitudes1, demands1, 5, [1000, 1000, 1000, 1000, 1000], 0, 15)
print(results)