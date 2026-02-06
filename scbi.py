import torch
import torch.nn as nn

def compute_scbi_weights(X_data, y_data, n_samples=10, sample_ratio=0.5, ridge_alpha=1.0):
    """
    Stochastic Covariance-Based Initialization (SCBI) for Regression or Binary Classification.
    
    Computes the optimal linear weights by solving the Normal Equation on stochastic subsets (Bagging).
    
    Args:
        X_data (torch.Tensor): Input features of shape [N, D].
        y_data (torch.Tensor): Target values of shape [N, 1].
        n_samples (int): Number of bagging subsets to average.
        sample_ratio (float): Fraction of data to use in each subset (0.0 to 1.0).
        ridge_alpha (float): Regularization strength to prevent singular matrices.
        
    Returns:
        weights (torch.Tensor): Shape [D, 1]
        bias (torch.Tensor): Shape [1]
    """
    n_total, n_features = X_data.shape
    subset_size = int(n_total * sample_ratio)
    
    accumulated_weights = torch.zeros(n_features, 1, device=X_data.device)
    accumulated_bias = torch.zeros(1, device=X_data.device)
    
    print(f"🚀 Running SCBI (Regression) on {n_samples} subsets...")
    
    for i in range(n_samples):
        # 1. Stochastic Sampling (Bagging)
        indices = torch.randperm(n_total, device=X_data.device)[:subset_size]
        X_sub = X_data[indices]
        y_sub = y_data[indices]
        
        # 2. Add Bias Column (Augmentation)
        # We append a column of 1s to solve for weights and bias simultaneously
        ones = torch.ones(subset_size, 1, device=X_data.device)
        X_sub_bias = torch.cat([X_sub, ones], dim=1) 
        
        # 3. Compute Interaction Matrix (Covariance): X^T * X
        XT_X = torch.matmul(X_sub_bias.T, X_sub_bias)
        
        # 4. Apply Ridge Regularization
        identity = torch.eye(XT_X.shape[0], device=X_data.device) * ridge_alpha
        XT_X_reg = XT_X + identity
        
        # 5. Compute Marginal Correlations: X^T * y
        XT_y = torch.matmul(X_sub_bias.T, y_sub)
        
        # 6. Solve Linear System
        # We use linalg.solve (or pinv fallback) for stability
        try:
            w_solved = torch.linalg.solve(XT_X_reg, XT_y)
        except RuntimeError:
            w_solved = torch.matmul(torch.linalg.pinv(XT_X_reg), XT_y)
            
        # 7. Separate Weights and Bias
        weights = w_solved[:-1]
        bias = w_solved[-1]
        
        accumulated_weights += weights
        accumulated_bias += bias
        
    # 8. Ensemble Averaging
    final_weights = accumulated_weights / n_samples
    final_bias = accumulated_bias / n_samples
    
    return final_weights, final_bias


def compute_scbi_classification(X_data, y_one_hot, n_samples=10, sample_ratio=0.5, ridge_alpha=1.0):
    """
    Stochastic Covariance-Based Initialization (SCBI) for Multi-Class Classification.
    
    Args:
        X_data (torch.Tensor): Input features of shape [N, D].
        y_one_hot (torch.Tensor): One-Hot Encoded targets of shape [N, C] where C is num_classes.
        n_samples (int): Number of bagging subsets.
        sample_ratio (float): Fraction of data per subset.
        ridge_alpha (float): Regularization strength.
        
    Returns:
        weights (torch.Tensor): Shape [D, C]
        bias (torch.Tensor): Shape [1, C]
    """
    n_total, n_features = X_data.shape
    n_classes = y_one_hot.shape[1]
    subset_size = int(n_total * sample_ratio)
    
    accumulated_weights = torch.zeros(n_features, n_classes, device=X_data.device)
    accumulated_bias = torch.zeros(1, n_classes, device=X_data.device)
    
    print(f"🚀 Running SCBI (Multi-Class) on {n_samples} subsets...")
    
    for i in range(n_samples):
        # 1. Sampling
        idx = torch.randperm(n_total, device=X_data.device)[:subset_size]
        X_sub = X_data[idx]
        y_sub = y_one_hot[idx]
        
        # 2. Add Bias Column
        ones = torch.ones(X_sub.shape[0], 1, device=X_data.device)
        X_sub_bias = torch.cat([X_sub, ones], dim=1)
        
        # 3. Interaction Matrix
        XT_X = torch.matmul(X_sub_bias.T, X_sub_bias)
        identity = torch.eye(XT_X.shape[0], device=X_data.device) * ridge_alpha
        XT_X_reg = XT_X + identity
        
        # 4. Correlation Matrix (Features vs Each Class)
        XT_y = torch.matmul(X_sub_bias.T, y_sub)
        
        # 5. Solve
        try:
            w_solved = torch.linalg.solve(XT_X_reg, XT_y)
        except RuntimeError:
            w_solved = torch.matmul(torch.linalg.pinv(XT_X_reg), XT_y)
            
        weights = w_solved[:-1]
        bias = w_solved[-1]
        
        accumulated_weights += weights
        accumulated_bias += bias
        
    return accumulated_weights / n_samples, accumulated_bias / n_samples


def compute_fast_damping_weights(X, y):
    """
    Linear Complexity Approximation: Correlation Damping.
    Use this when D > 10,000 and matrix inversion is impossible.
    
    Formula: w_i = Slope(x_i, y) / (1 + Sum(|Corr(x_i, x_j)|))
    """
    # 1. Standardize inputs for Correlation calculation
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True) + 1e-8 
    X_norm = (X - X_mean) / X_std
    
    # 2. Numerator: Univariate Correlation/Slope with Target
    # (Simplified assumption: X is standardized)
    numerator = torch.matmul(X_norm.T, y) / X.shape[0]
    
    # 3. Denominator: Damping Factor
    # Calculate Correlation Matrix (Z^T * Z) / N
    corr_matrix = torch.matmul(X_norm.T, X_norm) / X.shape[0]
    
    # Zero out diagonal (Self-correlation doesn't dampen)
    mask = ~torch.eye(corr_matrix.shape[0], dtype=torch.bool, device=X.device)
    neighbor_correlations = corr_matrix * mask
    
    # Sum of absolute neighbor correlations
    damping = 1.0 + torch.sum(torch.abs(neighbor_correlations), dim=1, keepdim=True)
    
    return numerator / damping