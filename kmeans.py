import numpy as np
import matplotlib.pyplot as plt

def get_data(total, seed=None):
    if seed != None:
        np.random.seed(2)
    
    # x, y
    return np.random.rand(total, 2)

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

data = get_data(100, seed=2)
centres = np.random.permutation(data)[:2]
data_with_dists = calc_dists(data, centres)
cluster_1, cluster_2 = get_clusters(data_with_dists)

print(cluster_1[0:5,:])
print(cluster_2[0:5,:])

#plt.scatter(data[:,0], data[:,1])
#plt.show()
