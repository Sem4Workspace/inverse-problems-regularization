import numpy as np
from tikhonov import tikhonov_reconstruction

def sweep_lambda(A, y, x_true, lambdas):
    residual_norms = []
    solution_norms = []
    errors = []

    for lam in lambdas:
        x_lam = tikhonov_reconstruction(A, y, lam)

        residual_norms.append(np.linalg.norm(A @ x_lam - y))
        solution_norms.append(np.linalg.norm(x_lam))
        errors.append(np.linalg.norm(x_lam - x_true))

    return residual_norms, solution_norms, errors
