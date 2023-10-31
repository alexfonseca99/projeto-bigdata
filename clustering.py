from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from settings import *


def agglomerative_clustering(data, n, affinity='euclidean', linkage='ward'):

    #cluster_object = SpectralClustering(n_clusters=n, assign_labels='discretize', random_state=0).fit(data)
    cluster_object = AgglomerativeClustering(n_clusters=n, affinity=affinity, linkage=linkage, compute_full_tree=True)
    labels = cluster_object.fit_predict(data)
    return labels


def get_clusters_by_frame(label, n_clusters):
    lista_de_labels = []
    # Iterate over a sequence of numbers from 0 to 4
    for i in range(n_clusters):
        # In each iteration, add an empty list to the main list
        lista_de_labels.append([])

    for i, lab in enumerate(label):
        lista_de_labels[lab].append(i) #corrigir frames

    return lista_de_labels