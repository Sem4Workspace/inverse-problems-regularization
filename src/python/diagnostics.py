import numpy as np

def compute_diagnostics(S):
    return {
        "min_singular": float(np.min(S)),
        "max_singular": float(np.max(S)),
        "condition_number": float(np.max(S) / np.min(S)),
        "spectral_decay_ratio": float(S[0] / S[-1])
    }
