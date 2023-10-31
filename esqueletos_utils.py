import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from settings import *
from main import *


def extract_coords(matrix):
    # so funciona para matrizes que tenham o frame, probabilidades e coordenadas
    # tira o numero do frame e as probabilidades e retorna so as coords
    x_coord = [i for i in range(1, matrix.shape[0], 3)]
    y_coord = [i for i in range(2, matrix.shape[0], 3)]
    coords_idx = x_coord + y_coord
    coords_idx.sort()

    return matrix[coords_idx, :]


def find_outliers_points(esqueletos):
  esq_com_outliers = []

  for k in range(esqueletos.shape[0]):
    esq = esqueletos[k, :]

    for i in range(0,len(esq), 2):

      if not (-2.5 < esq[i] < 2.5 and -2.5 < esq[i+1] < 2.5): #numa distribuição normal, se o |zscore|>3 o ponto é outlier
        esq_com_outliers.append(k) #este esqueleto tem um outlier
        esq[i] = np.nan #o ponto é invalidado (coordenada x)
        esq[i+1] = np.nan # "" " " (coordenada y)

    esqueletos[k, :] = esq

  return esq_com_outliers, esqueletos


def remove_outliers(outliers, base):
  out_tpl = np.nonzero(outliers)

  to_delete = []

  clean_list = []

  for i in range(base.shape[1]):
    if i not in out_tpl[0]: #é um inlier, pode ir para a clean list
      clean_list.append(base[:, i])
    # else:# é um outlier

  clean_list = np.array(clean_list)

  clean_list = np.transpose(clean_list)


  return clean_list


def detect_outliers(esqueletos, rank, u_in, s_in, v_in):
    inliers_indexes = [i for i in range(esqueletos.shape[1])]
    outliers_indexes = []

    base = esqueletos  # a base começa igual a todas as features

    n_iteracoes = 1

    outliers = []
    for i in range(n_iteracoes):
        outliers.append([])

    #######################################################################################################

    for i in range(n_iteracoes):
        THRESHOLD = 0.6
        # fazer svd da base
        if i == 0:
            u = u_in
            s = s_in
            v = v_in
        else:
            u, s, v = np.linalg.svd(base)

        # Pi
        Pi = u[:, :rank] @ u[:, :rank].T

        # PiN
        PiN = np.eye(base.shape[0]) - Pi

        # projections
        fi = Pi @ esqueletos
        fn = PiN @ esqueletos
        # normas
        di = np.linalg.norm(fi, axis=0)
        dn = np.linalg.norm(fn, axis=0)
        df = np.linalg.norm(esqueletos, axis=0)
        # ratios
        ri = np.true_divide(di, df)
        rnull = np.true_divide(dn, df)
        mean = np.mean(rnull)
        std = np.std(rnull)
        # plot dos ratios
        if NORM_VERBOSE:
            plt.hist(rnull, bins=21)
            plt.title("Histograma de normas para null (esqueletos)")
            plt.xlabel("Norma")
            plt.ylabel("Frequência")
            plt.show()
            plt.plot([i for i in range(len(ri))], ri)
            plt.title(f"Norma para a base (esqueletos)")
            plt.xlabel("Esqueleto")
            plt.ylabel("Norma")
            plt.show()
            plt.plot([i for i in range(len(rnull))], rnull, label="Normas")
            plt.plot([i for i in range(len(rnull))], [THRESHOLD for i in range(len(rnull))], label=r"Threshold")
            plt.legend()
            plt.title(f"Norma para null (esqueletos)")
            plt.xlabel("Esqueleto")
            plt.ylabel("Norma")
            plt.show()

        # encontrar outliers

        outliers[i] = (rnull > THRESHOLD) * rnull


        # remover outliers -> base = base - outliers
        if np.count_nonzero(outliers[i]) != 0:  # if there are outliers
            base = remove_outliers(outliers[i], esqueletos)
        if i != 0:
            if np.count_nonzero(outliers[i]) <= np.count_nonzero(outliers[i - 1]):  # no more changes to be made
                break
    percentage = np.count_nonzero(outliers[-1]) / esqueletos.shape[1] * 100
    print(f"Outlier percentage (skel): {percentage}%")
    return base, outliers, i


def rank_detection(s_in):
  cumulative_sum = np.cumsum(s_in)

  rank = next(x for x, val in enumerate(cumulative_sum/cumulative_sum[-1]) if val > 0.9)
  if SV_VERBOSE:
      plt.plot([i for i in range(0, len(s_in))], s_in)
      plt.title("Magnitude dos valores singulares (esqueletos)")
      plt.xlabel("Valor singular")
      plt.ylabel("Magnitude")
      plt.show()

  #print("rank for cumsum > 0.9", rank)

  threshold_estabilizaçao = 0.05

  for i in range(len(cumulative_sum)):

    if (cumulative_sum[i+1] - cumulative_sum[i])/cumulative_sum[i] < threshold_estabilizaçao: #considera-se que a reta estabiliza quando o proximo valor não melhora a soma x% da previous soma
      print(f"stabilized rank {i}")
      return i


def reduce_dimension(esqueletos, rank, u, s):

  projector = u[:, :rank] @ np.diag(s[:rank])

  projection = projector.T @ esqueletos

  projection = normalize(projection, axis=1, norm='l2')

  return projection


def count_n_esq(esqueletos, features):

  n_frames = features.shape[1]
  n_esq_frame = [0 for i in range(n_frames)]

  for i in range(esqueletos.shape[1]):
    esq = esqueletos[:, i]
    n_esq_frame[int(esq[0])] += 1

  index_nesq = []
  for x in range(np.max(n_esq_frame)+1):
      innerlist = []
      for y in range(n_frames):
        if x == n_esq_frame[y]:
          innerlist.append(y)
      index_nesq.append(innerlist)

  return n_esq_frame, index_nesq


def condensar_esqueletos(n_esq_frame, projection):
    esq_feat = []

    for k in range(len(n_esq_frame)):

      esq = np.zeros(projection.shape[0])

      if n_esq_frame[k] != 0:
        start = 0
        if k != 0:
          start = np.sum(n_esq_frame[:k])
        end = start + n_esq_frame[k]

        for j in range(start, end):
          esq = esq + projection[:, j]

        esq = esq/(end - start)

      esq_feat.append(esq)

    esq_feat = np.array(esq_feat)

    return esq_feat
