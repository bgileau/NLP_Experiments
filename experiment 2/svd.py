import numpy as np


class SVD(object):
    def __init__(self):
        pass

    def svd(self, data):
        U,S,V = np.linalg.svd(data)
        return U, S, V

    def rebuild_svd(self, U, S, V, k):
        data_rebuild = np.matmul(np.matmul(U[:, :k], np.diag(S[:k])), V[:k, :])
        return data_rebuild

    def compression_ratio(self, data, k):  # [5pts]
        num_stored_original = data.shape[0] * data.shape[1]
        num_stored_compressed = k * data.shape[0] + k + k * data.shape[1]
        compression_ratio = num_stored_compressed / num_stored_original
        return compression_ratio

    def recovered_variance_proportion(self, S, k):  # [5pts]
        variance_all = np.sum(S ** 2)
        variance_top_k = np.sum(S[:k] ** 2)
        recovered_var = variance_top_k / variance_all
        return recovered_var