"""
Projeto PBDat 21/22 - Grupo 2:
- José Pedro Marques, 89916
- Miguel Figueira, 90144
- Alexandre Fonseca, 90210
"""

from esqueletos import *
from features import *
from visualizations import *
from clustering import *
from scipy.io import loadmat
import argparse
from settings import *


def main(features_file, incomplete_esq_file, video_path):

    features = loadmat(features_file)['features']
    incomplete_esq = loadmat(incomplete_esq_file)['skeldata']
    incomplete_esq[0, :] += 1 #sincronizar frames com esqueletos

    if EDA_VERBOSE:
        complete_points_per_skeleton(incomplete_esq)

    #Processar dados
    print("Começar processamento\n")
    low_dim_features = process_features(features)
    low_dim_esq, n_esq_frame = process_esqueletos(incomplete_esq, features)
    low_dim_features = low_dim_features.T
    print("Acabar processamento\n")

    #Check dims
    #print(f"low_dim_features.shape={low_dim_features.shape}")
    #print(f"low_dim_esq.shape={low_dim_esq.shape}")

    #Stack features / Extra features
    stacked_features = np.hstack((low_dim_features, low_dim_esq))
    n_esq_frame_normalizado = (n_esq_frame - np.mean(n_esq_frame)) / np.std(n_esq_frame)
    extra_features = np.hstack((low_dim_features, n_esq_frame_normalizado[:, None]))


    #Absolutamente todas as features: image embeddings, poses, n_esq e numero do frame
    frames = [i for i in range(low_dim_features.shape[0])]
    frames_normalizado = (frames - np.mean(frames)) / np.std(frames)

    print("embedd:", low_dim_features.shape)
    print("skel:", low_dim_esq.shape)
    print("stacked:", stacked_features.shape)
    print("extra:", extra_features.shape)

    #Dendrogramas
    print("Começar dendrogramas\n")
    features_dendrogram = compute_dendrogram(low_dim_features)
    esq_dendrogram = compute_dendrogram(low_dim_esq)
    stacked_dendrogram = compute_dendrogram(stacked_features)
    extra_dendrogram = compute_dendrogram(extra_features)
    print("Acabar dendrogramas\n")

    #Clustering
    print("Começar clustering\n")
    n_feature_clusters = 3
    n_esq_clusters = 2 # ou 3
    n_stacked_clusters = 3
    n_extra_clusters = 3

    features_clusters = agglomerative_clustering(low_dim_features, n_feature_clusters)
    features_clusters = get_clusters_by_frame(features_clusters, n_feature_clusters)

    esq_clusters = agglomerative_clustering(low_dim_esq, n_esq_clusters)
    esq_clusters = get_clusters_by_frame(esq_clusters, n_esq_clusters)

    stacked_clusters = agglomerative_clustering(stacked_features, n_stacked_clusters)
    stacked_clusters = get_clusters_by_frame(stacked_clusters, n_stacked_clusters)

    extra_clusters = agglomerative_clustering(extra_features, n_extra_clusters)
    extra_clusters = get_clusters_by_frame(extra_clusters, n_extra_clusters)

    print("Acabar clustering\n")

    n_images = 5
    features_dists = closest_furthest_frame(low_dim_features, features_clusters,n_images)
    esq_dists = closest_furthest_frame(low_dim_esq, esq_clusters, n_images)
    stacked_dists = closest_furthest_frame(stacked_features, stacked_clusters, n_images)
    extra_dists = closest_furthest_frame(extra_features, extra_clusters, n_images)
    #Visualizations
    #compute_dendrogram, compute_tsne, show frames

    if video_path is not None:

        if CLUSTER_SAMPLE_VERBOSE:
            print("**** start embeddings ****")
            show_cluster_example(n_feature_clusters, SAMPLE_SIZE, features_clusters, video_path, "embedding only")

        if CLUSTER_SAMPLE_VERBOSE:
            print("**** start skel ****")
            show_cluster_example(n_esq_clusters, SAMPLE_SIZE, esq_clusters, video_path, "skel only")

        if CLUSTER_SAMPLE_VERBOSE:
            print("**** start embeddings + skel ****")
            show_cluster_example(n_stacked_clusters, SAMPLE_SIZE, stacked_clusters, video_path, "embedding + skel")

        if CLUSTER_SAMPLE_VERBOSE:
            print("**** start  embeddings + number of skel ****")

            show_cluster_example(n_extra_clusters, SAMPLE_SIZE, extra_clusters, video_path, "embeddings + number of skel")

    if TSNE_VERBOSE:
        compute_tsne(low_dim_features, "TSNE Embeddings", features_clusters)
        compute_tsne(low_dim_esq, "TSNE Esqueletos", esq_clusters)
        compute_tsne(stacked_features, "TSNE Embeddings + Esqueletos", stacked_clusters)
        compute_tsne(extra_features, "TSNE Embeddings + N_esqueletos", extra_clusters)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="file paths")

    parser.add_argument('--skel', type=str, help="incomplete skeleton mat file", required=True)
    parser.add_argument('--features', type=str, help="features mat file", required=True)
    parser.add_argument("--video", default=None)

    args = parser.parse_args()

    main(args.features, args.skel, args.video)