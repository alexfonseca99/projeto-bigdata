import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import cv2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import math
from settings import *


def compute_dendrogram(matrix, p=10):
    sys.setrecursionlimit(10000)
    dend = dendrogram(linkage(matrix, method='ward'), p=5, truncate_mode='level')
    plt.show()
    return dend


def show_frames(a_verificar, video_path, max_frames, cluster=None, mode=None):
    cap = cv2.VideoCapture(video_path)
    counter = 0
    seen = 0

    while (cap.isOpened() and seen < max_frames):
        ret, frame = cap.read()
        if counter in a_verificar:
            # images_load.append(frame)
            if cluster is None:
                cv2.imshow(f"frame {counter}", frame)
                cv2.imwrite(f'frame{counter}.jpg', frame)
            else:
                cv2.imshow(f"frame {counter} in cluster {cluster} - {mode}", frame)
            seen += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #if counter == 500:
            #break

    cap.release()
    cv2.destroyAllWindows()

    return


def compute_tsne(features, title, clusters):

    X_embedded = TSNE(n_components=2, init='random').fit_transform(features)
    for i in range(len(clusters)):
        plt.scatter(X_embedded[clusters[i], 0], X_embedded[clusters[i], 1], s=4, label=f"Cluster {i}")
    plt.title(title)
    plt.legend()
    plt.show()

    return


def show_cluster_example(n_clusters, n_images, clusters, video_path, mode):

    for i in range(n_clusters):
        show_frames(np.random.choice(clusters[i], n_images), video_path, n_images, cluster=i, mode=mode)

    return


#deve receber o esqueleto desnormalizado, sem frame e sem prob

def plot_belos_esq(esqueleto, a_plotar):
  # esqueleto = extract_coords(esqueletos_not_im_size)

  # esqueleto = reshape_im_size(esqueleto)

  esqueleto = esqueleto[:, a_plotar]


  print(esqueleto.shape)

  # for k in range(esqueleto.shape[1]):
  #   esq = esqueleto[:, k]
  #   print(esq)
  #   plt.scatter([esq[i] for i in range(0, 36, 2)] , [esq[i+1] for i in range(0, 36, 2)] )

  #   plt.xlim(0,640)
  #   plt.ylim(0,360)
  #   plt.show()

  for k in range(esqueleto.shape[1]):
    esq = esqueleto[:, k]
    x = [ esq[i] for i in range(0, len(esq), 2)]
    y = [ esq[i] for i in range(1, len(esq), 2)]

    extremidade_1 = [4, 3, 2, 16, 14, 15, 17, 1, 1, 1, 8, 9, 1, 11, 12, 5, 6]

    extremidade_2 = [3, 2, 1, 14, 0, 0, 15, 0, 5, 8, 9, 10, 11, 12, 13, 6, 7]

    #len(extremidade_1)
    for i in range(0, len(extremidade_1)):
      plt.plot( (x[extremidade_1[i]], x[extremidade_2[i]]), (y[extremidade_1[i]], y[extremidade_2[i]]),  'ro-')

    plt.xlim(0,640)
    plt.ylim(0,360)
    plt.gca().invert_yaxis()
    plt.show()


def reshape_im_size(esqueletos):
    for k in range(esqueletos.shape[1]):
        esq = esqueletos[:, k]

        if esqueletos.shape[0] == 55:
            for i in range(0, 54, 3):
                esq[1 + i] = int(esq[1 + i] * 640)
                esq[1 + i + 1] = int(esq[1 + i + 1] * 360)

        elif esqueletos.shape[0] == 36:
            for i in range(0, 36, 2):
                if math.isnan(esq[i]) or math.isnan(esq[i + 1]):
                    continue
                esq[i] = int(esq[i] * 640)
                esq[i + 1] = int(esq[i + 1] * 360)

        esqueletos[:, k] = esq

    return esqueletos


def complete_points_per_skeleton(esqueletos):

    n_pontos_por_esqueleto = [0 for i in range(esqueletos.shape[1])]
    #para cada esqueleto
    for k in range(esqueletos.shape[1]):
        esq = esqueletos[:, k]
        #conta-se o numero de pontos completos
        for x in range(1, 54, 3):
            if esq[x] != 0:
                n_pontos_por_esqueleto[k] += 1 #apanha-se o numero de pontos completos para cada esqueleto

    plt.hist(n_pontos_por_esqueleto, bins=np.max(n_pontos_por_esqueleto))
    plt.title("Histograma de número de pontos conhecidos por esqueleto")
    plt.xlabel("Número de pontos")
    plt.ylabel("Número de esqueletos")
    plt.show()

    return


def closest_furthest_frame(features, clusters_list, n):

    frame_list = [] #lista de listas. elementos são [closest, furthest]
    for i in range(len(clusters_list)):
        center = np.mean(features[clusters_list[i]], axis=0)

        dists = np.linalg.norm(features[clusters_list[i]] - center, axis=1)
        sorted_dists = np.argsort(dists)
        furthest = sorted_dists[-n:]
        closest = sorted_dists[:n]
        frame_list.append([closest, furthest])

    return frame_list

