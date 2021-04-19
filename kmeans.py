import sys

import numpy as np
import matplotlib.pyplot as plt

def get_data(points, seed=None):
    if seed != None:
        np.random.seed(2)
    
    data = np.random.rand(points, 2)
    stop_adding_i = (points - 1) // 2
    data[:stop_adding_i,:] += 0.75

    # x, y
    return data

def plot_data(data):
    plt.scatter(data[:,0], data[:,1])
    plt.show()

def calc_dists(data, centres):
    data_with_dists = data.copy()

    for i in range(centres.shape[0]):
        dists = np.sqrt(((data - centres[i]) ** 2).sum(axis=1))
        data_with_dists = np.column_stack((data_with_dists, dists))
    
    # x, y, d1, d2, ..., dn 
    return data_with_dists

def get_clusters(data_with_dists):
    cluster_1_mask = data_with_dists[:, 2] < data_with_dists[:, 3]
    cluster_2_mask = data_with_dists[:, 3] <= data_with_dists[:, 2]

    return data_with_dists[cluster_1_mask,:], data_with_dists[cluster_2_mask,:]

def calc_centres(cluster_1, cluster_2):
    centre_1 = cluster_1.mean(axis=0)[:2]
    centre_2 = cluster_2.mean(axis=0)[:2]
    
    return np.array((centre_1, centre_2))

def plot_clusters(cluster_1, cluster_2, centres):
    plt.scatter(cluster_1[:,0], cluster_1[:,1], color='blue')
    plt.scatter(cluster_2[:,0], cluster_2[:,1], color='red')
    plt.scatter(centres[0][0], centres[0][1], color='purple')
    plt.scatter(centres[1][0], centres[1][1], color='purple')
    plt.show()

def main(points, seed=None):
    data = get_data(points, seed=seed)
    curr_centres = np.random.permutation(data)[:2]
    prev_centres = None

    print('Plotting data...')
    plot_data(data)

    while not(np.array_equal(curr_centres, prev_centres)):
        data_with_dists = calc_dists(data, curr_centres)
        cluster_1, cluster_2 = get_clusters(data_with_dists)

        print('Plotting intermediate clusters...')
        plot_clusters(cluster_1, cluster_2, curr_centres)

        prev_centres = curr_centres
        curr_centres = calc_centres(cluster_1, cluster_2)

    print('Plotting final clusters...')
    plot_clusters(cluster_1, cluster_2, curr_centres)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        print('Starting k-means with a seed...')
        main(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print('Starting k-means without a seed...')
        main(int(sys.argv[1]))
    
    print('Completed execution.')
