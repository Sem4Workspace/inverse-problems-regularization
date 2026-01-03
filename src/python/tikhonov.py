import numpy as np

def tikhonov_reconstruction(A, y, lam):
    """
    Solves Tikhonov-regularized inverse problem:
        min ||Ax - y||^2 + lam ||x||^2

    Parameters
    ----------
    A : ndarray (m x n)
        Forward operator
    y : ndarray (m,)
        Observed data
    lam : float
        Regularization parameter (lambda)

    Returns
    -------
    x_lam : ndarray (n,)
        Regularized reconstruction
    """
    n = A.shape[1]
    ATA = A.T @ A
    ATy = A.T @ y

    x_lam = np.linalg.solve(ATA + lam * np.eye(n), ATy)
    return x_lam
