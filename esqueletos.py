from esqueletos_utils import *
from sklearn.preprocessing import StandardScaler
import h2o
from h2o.estimators import H2OGeneralizedLowRankEstimator
import numpy as np
from settings import *


def process_esqueletos(incomplete_esqueletos, features, D_original = None):
    np.seterr(invalid='ignore')
    n_esq_frame, index_nesq = count_n_esq(incomplete_esqueletos, features)
    incomplete_esqueletos = extract_coords(incomplete_esqueletos)
    incomplete_esqueletos[incomplete_esqueletos == 0] = np.nan

    incomplete_esqueletos = incomplete_esqueletos.T
    scaler = StandardScaler()
    incomplete_esqueletos = scaler.fit_transform(incomplete_esqueletos)

    #esq_com_outliers, incomplete_esqueletos = find_outliers_points(incomplete_esqueletos)
    # Subtrair media e dividir por desvio padrão


    if D_original is None:
        # Converter para data frame
        h2o.init()
        incomplete_esqueletos_df = h2o.H2OFrame(incomplete_esqueletos)

        # Build and train the model:
        glrm_model = H2OGeneralizedLowRankEstimator(k=11,
                                                    loss="quadratic",
                                                    gamma_x=0.5,
                                                    gamma_y=0.5,
                                                    max_iterations=1500,
                                                    recover_svd=True,
                                                    init="SVD",
                                                    impute_original=False,
                                                    transform="none")

        glrm_model.train(training_frame=incomplete_esqueletos_df)
        D_original = glrm_model.predict(incomplete_esqueletos_df).as_data_frame(use_pandas=True).to_numpy()

    D = np.copy(D_original)

    # Reverter a normalização
    D = scaler.inverse_transform(D)
    #para as dimensões da imagem
    D[:, 0:-1:2] *= 640
    D[:, 1::2] *= 360
    D = np.round(D)

    # remoção dos outliers da base
    u_in, s_in, v_in = np.linalg.svd(D_original.T)
    rank = rank_detection(s_in)
    base, outliers, ultima_iter = detect_outliers(D_original.T, rank, u_in, s_in, v_in)

    # A BASE È RETORNADA COM DIMENSOES rank por n_esqueletos !!!!

    # redução da dimensão
    u_in, s_in, v_in = np.linalg.svd(base)
    rank = rank_detection(s_in)
    projection = reduce_dimension(D.T, rank, u_in, s_in) #faz sentido projetar o D porque tem informação espacial

    #  condensar os esqueletos num so frame
    esq_feat = condensar_esqueletos(n_esq_frame, projection)
    h2o.cluster().shutdown()

    return esq_feat, n_esq_frame