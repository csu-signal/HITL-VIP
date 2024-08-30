# USAGE (assumes you're in python_vip folder)
# run sh ../evaluation/batch_file_process.sh [--pilots_only -p] to make the arrays file
# run python ../evaluation/clusters.py -f ../evaluation/arrays.txt

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import decomposition

def main():
    parser = argparse.ArgumentParser(
                    prog='clusters',
                    description='Cluster of PyVIP analysis features')
    parser.add_argument('-f', '--filename')
    args = parser.parse_args()
    print(f"Opening {args.filename}")
    
    # load saved arrays
    arr = np.loadtxt(args.filename, dtype=np.float32)
    print(arr)
    print(arr.shape)
    
    features = np.hstack([arr[:, 1:]])
    
    means = np.mean(features,axis=0)
    sds = np.std(features,axis=0)
    features_standardized = (features-means)/sds

    # cluster the unreduced data
    #clustering = AgglomerativeClustering(n_clusters=3).fit(features_standardized)
    clustering = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(features_standardized)
    print(clustering.labels_)
    
    # fit 3D PCA
    pca = decomposition.PCA(n_components=3)
    pca.fit(features_standardized)
    X = pca.transform(features_standardized)
    
    # plotting
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    
    ax = fig.add_subplot(111, projection="3d", elev=45, azim=135)
    ax.set_position([0, 0, 0.95, 1])
    plt.cla()
    
    colors = ["red", "blue", "green", "orange", "black"]
    edgecolors = [colors[i] for i in clustering.labels_]
    markercolors = [colors[i] for i in np.array(arr[:,0],dtype=int)]
    
    # label trial number (black) and # crashes (red)
    # color marker according to the cluster
    # outline marker according to the proficiency
    #   of model training data
    num_crashes = np.array(features[:,0]*30,dtype=int)
    print(num_crashes)
    print(np.array(arr[:,0],dtype=int))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=30, c=markercolors, edgecolors=edgecolors, linewidth=2, alpha=1)
    for i, _ in enumerate(X):
        ax.text(X[i, 0], X[i, 1], X[i, 2], i+1, 'x')
        ax.text(X[i, 0], X[i, 1], X[i, 2]-.1, num_crashes[i], (1,0,0), color='red')
    
    plt.show()

if __name__ == "__main__":
    main()
