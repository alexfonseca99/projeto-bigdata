from features_utils import *
import numpy as np
from settings import *
from sklearn.preprocessing import StandardScaler

def process_features(features):
    scaler = StandardScaler()
    features = scaler.fit_transform(features.T)
    features = features.T
    u_in, s_in, v_in = np.linalg.svd(features)
    rank = rank_detection(s_in)

    base, outliers, iter_final = detect_outliers(features, rank, u_in, s_in, v_in)
    out_tpl = np.nonzero(outliers[iter_final])
    outliers_indexes = out_tpl[0]

    #  definir rank da base (base é o conjunto de esqueletos que representa todos os outros)
    u_in, s_in, v_in = np.linalg.svd(base)
    #rank = rank_detection(s_in)
    #  reduzir dimensões
    projection = reduce_dimension(features, rank, u_in, s_in)

    return projection


