import numpy as np
import pandas
import json

def initialise_matrix(dist_file):


    dist = np.load(dist_file)

    for i in range(0, len(dist)):
        dist[i][i] = np.nan
        for j in range(i+1, len(dist)):
            if dist[i][j] == 1:
                dist[i][j] = 0.99
                dist[j][i] = 0.99

    return np.asmatrix(dist)

def initialise_headings(heading_file):

    with open(heading_file) as heading:
        headings = json.loads(heading.read())

    temp_dict = {}

    for i in range(0, len(headings)):
        temp_dict[i] = headings[i]

    return temp_dict

def get_matrix_min(dist):

    min_arr = np.where(dist == np.nanmin(dist))

    return min_arr[0]
