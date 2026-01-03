import numpy as np

def tsvd_reconstruction(U, S, Vt, y, k):
    Uk = U[:, :k]
    Sk = S[:k]
    Vk = Vt[:k, :]

    x_k = Vk.T @ np.diag(1 / Sk) @ Uk.T @ y
    return x_k
