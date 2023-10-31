import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from settings import *
from main import *

def remove_outliers(outliers, base):

    out_tpl = np.nonzero(outliers)

    to_delete = []

    clean_list = []

    for i in range(base.shape[1]):
        if i not in out_tpl[0]:  # é um inlier, pode ir para a clean list
            clean_list.append(base[:, i])
        # else:# é um outlier

    clean_list = np.array(clean_list)
    clean_list = np.transpose(clean_list)

    return clean_list


def detect_outliers(features_load, rank, u_in, s_in, v_in):
    inliers_indexes = [i for i in range(features_load.shape[1])]
    outliers_indexes = []

    base = features_load  # a base começa igual a todas as features

    n_iteracoes = 1

    outliers = []
    for i in range(n_iteracoes):
        outliers.append([])

    #######################################################################################################

    for i in range(n_iteracoes):
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
        fi = Pi @ features_load
        fn = PiN @ features_load
        # normas
        di = np.linalg.norm(fi, axis=0)
        dn = np.linalg.norm(fn, axis=0)
        df = np.linalg.norm(features_load, axis=0)
        # ratios
        ri = np.true_divide(di, df)
        rnull = np.true_divide(dn, df)
        mean = np.mean(rnull)
        std = np.std(rnull)
        # plot dos ratios
        if NORM_VERBOSE:
            plt.hist(rnull, bins=21)
            plt.title("Histograma de normas para null (embeddings)")
            plt.xlabel("Norma")
            plt.ylabel("Frequência")
            plt.show()
            plt.plot([i for i in range(len(ri))], ri)
            plt.title(f"Norma para a base (embeddings)")
            plt.xlabel("Frame")
            plt.ylabel("Norma")
            plt.show()
            plt.plot([i for i in range(len(rnull))], rnull)
            plt.plot([i for i in range(len(rnull))], [mean + 2 * std for i in range(len(rnull))], label=r"2$\sigma$")
            plt.plot([i for i in range(len(rnull))], [mean - 2 * std for i in range(len(rnull))], label=r"-2$\sigma$")
            plt.legend()
            plt.title(f"Norma para kernel (embeddings)")
            plt.xlabel("Frame")
            plt.ylabel("Norma")
            plt.show()

        # encontrar outliers

        rnull_aux = (rnull - mean) / std
        outliers_aux1 = np.logical_not(np.array((rnull_aux < 2)))
        outliers_aux2 = np.logical_not(np.array((rnull_aux > -2)))
        outliers[i] = np.logical_or(outliers_aux1, outliers_aux2) * rnull


        # remover outliers -> base = base - outliers
        if np.count_nonzero(outliers[i]) != 0:  # if there are outliers
            base = remove_outliers(outliers[i], features_load)
        if i != 0:
            if np.count_nonzero(outliers[i]) <= np.count_nonzero(outliers[i - 1]):  # no more changes to be made
                break
    percentage = np.count_nonzero(outliers[i]) / features_load.shape[1] * 100
    print(f"Outlier percentage (embeddings): {percentage}%")
    return base, outliers, i


def rank_detection(s_in):

    cumulative_sum = np.cumsum(s_in)

    rank = next(x for x, val in enumerate(cumulative_sum/cumulative_sum[-1]) if val > 0.9)
    if SV_VERBOSE:
        plt.plot([i for i in range(0, len(s_in))], s_in)
        plt.title("Singular Values")
        plt.title("Magnitude dos valores singulares (embeddings)")
        plt.xlabel("Valor singular")
        plt.ylabel("Magnitude")
        plt.show()

    #print("rank for cumsum > 0.9", rank)

    threshold_estabilizacao = 0.05

    for i in range(len(cumulative_sum)):
        if (cumulative_sum[i+1] - cumulative_sum[i])/cumulative_sum[i] < threshold_estabilizacao: #considera-se que a reta estabiliza quando o proximo valor não melhora a soma x% da previous soma
          print(f"stabilized rank {i}")
          return i #este é o rank


def reduce_dimension(features_load, rank, u, s):

    projector = u[:, :rank] @ np.diag(s[:rank])
    projection = projector.T @ features_load

    projection = normalize(projection, axis=1, norm='l2')

    return projection