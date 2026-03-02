import numpy as np
from geopy.distance import distance
from geopy import Point as GeoPoint
from pyproj import Geod
import json

_GEOD = Geod(ellps='WGS84')
_M_TO_MILES = 1.0 / 1609.344


def _read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def read_parameters_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    parameters = {item['parameter']: item['value'] for item in data}
    descriptions = {
        item['parameter']: {
            "description": item['description'],
            **({"choices": item['choices']} if 'choices' in item else {}),
        }
        for item in data
    }
    return parameters, descriptions


def _dist_matrix(locations):
    assert len(locations) > 1, "At least two points are required"
    m = len(locations)
    lats = np.array([loc[0] for loc in locations])
    lons = np.array([loc[1] for loc in locations])
    rows, cols = np.triu_indices(m, k=1)
    # Single vectorized C-level call over all upper-triangle pairs.
    # pyproj takes (lon, lat); result is meters → convert to miles.
    # Precision: agrees with geopy/geographiclib to <2e-12 miles (~3 nm).
    _, _, dist_m = _GEOD.inv(lons[rows], lats[rows], lons[cols], lats[cols])
    dist_miles = dist_m * _M_TO_MILES
    dist_matrix = np.zeros((m, m))
    dist_matrix[rows, cols] = dist_miles
    dist_matrix[cols, rows] = dist_miles
    return dist_matrix


def _generate_random_point(center, radius):
    r = radius * np.sqrt(np.random.rand())
    theta = np.random.uniform(0, 2 * np.pi)
    origin = GeoPoint(center)
    destination = distance(miles=r).destination(
        point=origin, bearing=np.degrees(theta)
    )
    return destination.latitude, destination.longitude


def extract_routes(x_matrix: np.array) -> list:
    """
    Extracts the routes followed by each vehicle based on the decision matrix.

    :param x_matrix: Binary decision matrix
    :return: List of routes (each route is a list of customer indices)
    """
    # Precompute successor of every node in one vectorized pass (O(n) argmax)
    # instead of one np.where scan per route step.
    next_arr = x_matrix.argmax(axis=1)
    first_visits = np.where(x_matrix[0] == 1)[0]
    num_vehicles = len(first_visits)
    routes = [[0, int(v)] for v in first_visits]
    for vehicle in range(num_vehicles):
        current = routes[vehicle][-1]
        while current != 0:
            current = int(next_arr[current])
            routes[vehicle].append(current)
    return routes
