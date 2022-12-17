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

    return (cities, demands, latitudes, longitudes)


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


'''matrix1 = create_distance_matrix(latitudes1, longitudes1)
for row in matrix1:
    print(row)'''


def create_vehicles(n_vehicles, capacities, depot_index):
    if n_vehicles != len(capacities):
        print('Nr of vehicles doesn\'t much length of capacities')
    vehicles = []
    for i in range(n_vehicles):
        vehicles.append([i, capacities[i], depot_index])

    return vehicles


'''vehicles1 = create_vehicles(5, [10, 10, 10, 10, 10])
for v in vehicles1:
    print(v)'''


def select_destination(distances, delivered, capacity, demands):
    min_distance = float('inf')
    min_index = float('inf')
    for i in range(len(distances)):
        if distances[i] < min_distance and distances.index(distances[i]) not in delivered and capacity >= demands[i]:
            min_distance = distances[i]
            min_index = distances.index(distances[i])

    if min_distance == float('inf'):
        min_distance = distances[0]
        min_index = 0

    return min_distance, min_index


#print(select_destination([0, 13, 4, 34, 8, 82, 81, 3, 22, 9], [0, 3, 4], 10, [1, 1, 11, 1, 1, 1, 1, 12, 1, 1]))


def deliver(latitudes, longitudes, demands, n_vehicles, capacities, depot_index):
    matrix = create_distance_matrix(latitudes, longitudes)
    original_capacities = capacities
    delivered = [0]

    vehicles = create_vehicles(n_vehicles, capacities, depot_index)
    vehicle_routes = {}
    for vehicle in vehicles:
        vehicle_routes[vehicle[0]] = {}
        vehicle_routes[vehicle[0]]['Locations'] = []
        vehicle_routes[vehicle[0]]['Distance'] = 0

    while len(delivered) != len(latitudes):
        for vehicle in vehicles:
            if len(delivered) == len(latitudes):
                break
            current_row = matrix[vehicle[2]]
            available_capacity = vehicle[1]
            distance, des_id = select_destination(current_row, delivered, available_capacity, demands)
            if des_id == depot_index:
                vehicle[1] = original_capacities[vehicle[0]]
            delivered.append(des_id)
            vehicle[2] = des_id
            vehicle_routes[vehicle[0]]['Distance'] += distance
            vehicle_routes[vehicle[0]]['Locations'].append(des_id)

    return vehicle_routes


def pprint(dictionary):
    for key, value in dictionary.items():
        print(f'{key}\n{value}\n')


results = deliver(latitudes1, longitudes1, demands1, 5, [1000, 1000, 1000, 1000, 1000], 0)
pprint(results)
