import numpy as np

def package_diagnostics(
    singular_values,
    condition_number,
    tikh_errors,
    lambdas,
    tsvd_errors,
    ks
):
    """
    Packages inverse-problem diagnostics for LLM reasoning.
    """

    diagnostics = {}

    # Spectral diagnostics
    diagnostics["condition_number"] = float(condition_number)
    diagnostics["spectral_decay_ratio"] = float(
        singular_values[0] / singular_values[-1]
    )

    # Singular value decay pattern
    decay = np.log10(singular_values + 1e-12)
    diagnostics["singular_decay_trend"] = (
        "rapid" if decay[0] - decay[-1] > 4 else "moderate"
    )

    # Tikhonov diagnostics
    diagnostics["tikhonov"] = {
        "lambdas": lambdas.tolist(),
        "errors": tikh_errors,
        "best_lambda": float(lambdas[np.argmin(tikh_errors)])
    }

    # TSVD diagnostics
    diagnostics["tsvd"] = {
        "ks": list(ks),
        "errors": tsvd_errors,
        "best_k": int(ks[np.argmin(tsvd_errors)])
    }

    return diagnostics
