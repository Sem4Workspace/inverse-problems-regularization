"""
Non-Stationary Iterated Tikhonov (NSIT) Regularization

NSIT improves the solution step by step using regularized corrections
of the residual, where the regularization strength decreases over iterations.

Mathematical Foundation:

NSIT update rule:
    x_n = x_{n-1} + (A^T*A + α_n*I)^{-1} * A^T * r_{n-1}

where:
    r_{n-1} = y_δ - A*x_{n-1}  (residual)
    α_n = decreasing regularization parameter

Morozov Stopping: Stop when ||y - Ax_n|| ≈ τ*δ*||y||
"""

import numpy as np
from numpy.linalg import svd, norm
    

def nsit_with_morozov(A: np.ndarray, y: np.ndarray, noise_level: float,
                      schedule_type: str = 'sqrt', tau: float = 1.0, 
                      max_iter: int = 100):
    """
    Non-Stationary Iterated Tikhonov (NSIT) with Morozov stopping.
    
    Iteratively refines solution using regularized corrections with
    decreasing regularization parameter. Stops automatically when residual
    reaches approximately τ*δ*||y||.
    
    Parameters:
    -----------
    A : np.ndarray
        Forward operator matrix (m × n)
    y : np.ndarray
        Measurement vector (length m)
    noise_level : float
        Relative noise level δ
    schedule_type : str, default='sqrt'
        Type of decreasing schedule: 'sqrt', 'linear', 'exp', or 'power'
    tau : float, default=1.0
        Safety factor for Morozov principle
    max_iter : int, default=100
        Maximum number of iterations
        
    Returns:
    --------
    x : np.ndarray
        Reconstructed solution
    history : dict
        Contains 'residuals': ||y - Ax_n||
                'alphas': regularization parameters used
                'stopping_iter': iteration where stopped
                'x': solution at each iteration
    """
    m, n = A.shape
    x = np.zeros(n)
    
    # Auto-select initial alpha
    U, s, _ = svd(A, full_matrices=False)
    alpha_0 = s.max() ** 2
    
    # Setup schedule function
    schedules = {
        'sqrt': lambda n: alpha_0 / np.sqrt(n + 1),
        'linear': lambda n: alpha_0 / (n + 1),
        'exp': lambda n: alpha_0 * (0.9 ** n),
        'power': lambda n: alpha_0 / ((n + 1) ** 1.5),
    }
    
    if schedule_type not in schedules:
        raise ValueError(f"schedule_type must be one of {list(schedules.keys())}")
    
    schedule = schedules[schedule_type]
    
    # Setup Morozov criterion
    target_residual = tau * noise_level * norm(y)
    
    # Precompute A^T*A for efficiency
    AtA = A.T @ A
    
    history = {
        'x': [x.copy()],
        'residuals': [],
        'alphas': [],
        'stopping_iter': max_iter - 1
    }
    
    # Main iteration loop
    for iter_count in range(max_iter):
        alpha_n = schedule(iter_count)
        
        # Compute residual
        r = y - A @ x
        residual_norm = norm(r)
        
        # Update: x_n = x_{n-1} + (A^T*A + α_n*I)^{-1} * A^T * r
        correction = np.linalg.solve(AtA + alpha_n * np.eye(n), A.T @ r)
        x = x + correction
        
        # Store history
        history['residuals'].append(residual_norm)
        history['alphas'].append(alpha_n)
        history['x'].append(x.copy())
        
        # Check Morozov stopping criterion
        if residual_norm <= target_residual:
            history['stopping_iter'] = iter_count
            break
    
    return x, history
