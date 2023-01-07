import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def normalize_weights(w_matrix):
    total = np.sum(w_matrix)
    for row in range(len(w_matrix)):
        for cell in range(len(w_matrix[row])):
            w_matrix[row, cell] = w_matrix[row, cell]/total

    return w_matrix


def update_weights(w_matrix, ants_routes, alpha=1.3, beta=0.9):
    w_matrix = w_matrix * beta

    for route in ants_routes:
        for i in range(len(route) - 1):
            w_matrix[route[i]][route[i+1]] = w_matrix[route[i]][route[i+1]] * alpha

    w_matrix = normalize_weights(w_matrix)

    return w_matrix


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
    index, distance = 0, distances[0]
    for i in range(len(weights)):
        distance = random.choices(distances, weights)[0]
        index = distances.index(distance)
        if index not in delivered and demands[index] <= capacity:
            break

    return index, distance


def deliver(latitudes, longitudes, demands, n_ants, capacities, start_index, n_groups=1, n_runs=1):
    d_matrix = create_distance_matrix(latitudes, longitudes)
    w_matrix = create_weights_matrix(len(latitudes), len(longitudes))
    original_capacities = capacities

    best_run = [float('inf'), []]
    group_bests = []
    for run in range(n_runs):

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
                    des_id, distance = select_destination(d_matrix[ant[2]], w_matrix[ant[2]], delivered, ant[1], demands)
                    delivered.append(des_id)
                    if des_id == start_index:
                        ant[1] = original_capacities[ant[0]]
                    ant[2] = des_id
                    ants_routes[ant[0]]['Locations'].append(des_id)
                    ants_routes[ant[0]]['Distance'] += distance

            for ant in ants:
                if ants_routes[ant[0]]['Locations'][-1] != 0:
                    d_to_startpoint = d_matrix[ant[2]][0]
                    ants_routes[ant[0]]['Locations'].append(start_index)
                    ants_routes[ant[0]]['Distance'] += d_to_startpoint

            # store group run results
            group_total_distance = 0
            for ant in ants_routes.keys():
                group_total_distance += ants_routes[ant]['Distance']

            if group_total_distance < best_group[0]:
                best_group[0] = group_total_distance
                best_group[1] = []
                for ant in ants_routes.keys():
                    best_group[1].append(ants_routes[ant]['Locations'])

        # update weights
        w_matrix = update_weights(w_matrix, best_group[1])

        print(f'Group best: {best_group[0]}')
        group_bests.append(best_group[0])
        if best_group[0] < best_run[0]:
            best_run[0] = best_group[0]
            best_run[1] = best_group[1]

    return best_run, group_bests


def pprint(dictionary):
    total_distance = 0
    for key, value in dictionary.items():
        print(f'{key}: {value}\n')
        total_distance += value['Distance']
    print(f'Total distance is {round(total_distance)} km.')


run_best, groups_bests = deliver(latitudes1, longitudes1, demands1, 5, [1000, 1000, 1000, 1000, 1000], 0, 10, 200)
plt.plot([i for i in range(len(groups_bests))], groups_bests)
plt.show()
print(run_best)