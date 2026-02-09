"""
SCBI: Stochastic Covariance-Based Initialization
A novel neural network weight initialization method.

Author: Fares Ashraf
Created: February 6, 2026
License: MIT

Citation:
    If you use this method in your research, please cite:
    Fares Ashraf. "Stochastic Covariance-Based Initialization for Neural Networks." 2026
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

__version__ = "1.0.0"
__author__ = "Fares Ashraf"
__all__ = [
    'SCBIInitializer',
    'FastDampingInitializer',
    'scbi_init',
    'fast_damping_init',
]


class SCBIInitializer:
    """
    Stochastic Covariance-Based Initialization (SCBI).
    
    A novel initialization method that computes optimal linear weights by:
    1. Stochastic bagging for robustness
    2. Solving the Normal Equation with ridge regularization
    3. Ensemble averaging across multiple subsets
    
    Key Innovations:
    - Universal formulation for regression and classification
    - Stochastic sampling prevents overfitting to training data
    - Ridge regularization ensures numerical stability
    - Linear complexity fallback for high-dimensional spaces
    
    Mathematical Foundation:
        For each subset: (X^T X + λI)^(-1) X^T y
        Final weights: Average over n_samples subsets
    
    Examples:
        >>> # Regression
        >>> initializer = SCBIInitializer(n_samples=10, ridge_alpha=1.0)
        >>> weights, bias = initializer.compute_weights(X_train, y_train)
        
        >>> # Classification
        >>> linear_layer = nn.Linear(50, 10)
        >>> initializer.initialize_layer(linear_layer, X_train, y_onehot)
    """
    
    def __init__(
        self,
        n_samples: int = 10,
        sample_ratio: float = 0.5,
        ridge_alpha: float = 1.0,
        verbose: bool = True
    ):
        """
        Initialize SCBI with hyperparameters.
        
        Args:
            n_samples: Number of stochastic subsets (higher = more stable, slower)
                      Recommended: 5-20 for most tasks
            sample_ratio: Fraction of data per subset (0, 1]
                         Recommended: 0.5 for balanced bias-variance tradeoff
            ridge_alpha: Regularization strength (higher = more regularization)
                        Recommended: 0.5-2.0, increase for noisy/high-dim data
            verbose: Print progress information
        """
        assert n_samples > 0, f"n_samples must be positive, got {n_samples}"
        assert 0 < sample_ratio <= 1, f"sample_ratio must be in (0, 1], got {sample_ratio}"
        assert ridge_alpha > 0, f"ridge_alpha must be positive, got {ridge_alpha}"
        
        self.n_samples = n_samples
        self.sample_ratio = sample_ratio
        self.ridge_alpha = ridge_alpha
        self.verbose = verbose
        
    def compute_weights(
        self,
        X_data: torch.Tensor,
        y_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SCBI initialization weights.
        
        Args:
            X_data: Input features [N, D]
            y_data: Targets [N] or [N, Output_Dim]
                   For regression: [N] or [N, 1]
                   For classification: [N, num_classes] (one-hot encoded)
            
        Returns:
            weights: [D, Output_Dim] - Weight matrix
            bias: [Output_Dim] - Bias vector
            
        Note:
            For PyTorch nn.Linear layers, use:
                layer.weight.data = weights.T
                layer.bias.data = bias
        """
        # Input validation and preprocessing
        assert X_data.ndim == 2, f"X_data must be 2D, got shape {X_data.shape}"
        
        X_data = X_data.float()
        if y_data.ndim == 1:
            y_data = y_data.unsqueeze(1)
        y_data = y_data.float()
        
        n_total, n_features = X_data.shape
        n_outputs = y_data.shape[1]
        subset_size = max(1, int(n_total * self.sample_ratio))
        
        # Initialize accumulators
        accumulated_weights = torch.zeros(
            n_features, n_outputs, 
            device=X_data.device, 
            dtype=X_data.dtype
        )
        accumulated_bias = torch.zeros(
            1, n_outputs, 
            device=X_data.device, 
            dtype=X_data.dtype
        )
        
        if self.verbose:
            task_type = "Regression" if n_outputs == 1 else f"Classification ({n_outputs} classes)"
            print(f"🚀 SCBI Initialization ({task_type})")
            print(f"   Subsets: {self.n_samples} × {subset_size} samples ({self.sample_ratio:.1%})")
            print(f"   Features: {n_features} → Outputs: {n_outputs}")
            print(f"   Ridge α: {self.ridge_alpha}")
        
        # Stochastic ensemble loop
        for i in range(self.n_samples):
            # 1. Stochastic bagging
            indices = torch.randperm(n_total, device=X_data.device)[:subset_size]
            X_sub = X_data[indices]
            y_sub = y_data[indices]
            
            # 2. Augment with bias column
            ones = torch.ones(
                X_sub.shape[0], 1, 
                device=X_data.device, 
                dtype=X_data.dtype
            )
            X_sub_bias = torch.cat([X_sub, ones], dim=1)
            
            # 3. Covariance matrix: X^T @ X
            XT_X = torch.matmul(X_sub_bias.T, X_sub_bias)
            
            # 4. Ridge regularization (differential for bias term)
            identity = torch.eye(
                XT_X.shape[0], 
                device=X_data.device, 
                dtype=X_data.dtype
            )
            identity[-1, -1] = 0.01  # Lighter regularization on bias
            XT_X_reg = XT_X + identity * self.ridge_alpha
            
            # 5. Correlation matrix: X^T @ y
            XT_y = torch.matmul(X_sub_bias.T, y_sub)
            
            # 6. Solve normal equation
            try:
                w_solved = torch.linalg.solve(XT_X_reg, XT_y)
            except RuntimeError:
                if self.verbose and i == 0:
                    print(f"   ⚠️  Matrix singular, using pseudo-inverse")
                w_solved = torch.matmul(torch.linalg.pinv(XT_X_reg), XT_y)
            
            # 7. Decompose weights and bias
            weights = w_solved[:-1]  # [D, Output_Dim]
            bias = w_solved[-1]      # [Output_Dim]
            
            accumulated_weights += weights
            accumulated_bias += bias
        
        # 8. Ensemble averaging
        final_weights = accumulated_weights / self.n_samples
        final_bias = accumulated_bias / self.n_samples
        
        # 9. Shape normalization
        if final_bias.ndim == 2 and final_bias.shape[0] == 1:
            final_bias = final_bias.squeeze(0)
        
        if self.verbose:
            print(f"✅ Initialization complete")
            print(f"   Weights: {final_weights.shape} | Bias: {final_bias.shape}")
        
        return final_weights, final_bias
    
    def initialize_layer(
        self,
        layer: nn.Linear,
        X_data: torch.Tensor,
        y_data: torch.Tensor
    ) -> None:
        """
        Initialize a PyTorch Linear layer in-place.
        
        Args:
            layer: nn.Linear layer to initialize
            X_data: Input features [N, D]
            y_data: Targets [N] or [N, Output_Dim]
            
        Example:
            >>> model = nn.Linear(50, 10)
            >>> initializer = SCBIInitializer()
            >>> initializer.initialize_layer(model, X_train, y_train)
        """
        assert isinstance(layer, nn.Linear), "Layer must be nn.Linear"
        
        weights, bias = self.compute_weights(X_data, y_data)
        
        with torch.no_grad():
            layer.weight.data = weights.T  # PyTorch stores as [out, in]
            layer.bias.data = bias


class FastDampingInitializer:
    """
    Fast Correlation Damping Initialization.
    
    Linear-complexity approximation of SCBI for high-dimensional problems.
    Uses correlation damping instead of matrix inversion.
    
    Complexity: O(N × D²) vs O(N × D³) for SCBI
    Use when: D > 10,000
    
    Formula:
        w_i = Corr(x_i, y) / (1 + Σ|Corr(x_i, x_j)|)
    
    The damping factor prevents overfitting by penalizing features
    that are highly correlated with many other features.
    
    Examples:
        >>> initializer = FastDampingInitializer()
        >>> weights, bias = initializer.compute_weights(X_train, y_train)
    """
    
    def __init__(self, eps: float = 1e-8, verbose: bool = True):
        """
        Initialize Fast Damping method.
        
        Args:
            eps: Numerical stability constant
            verbose: Print progress information
        """
        self.eps = eps
        self.verbose = verbose
    
    def compute_weights(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fast damping initialization weights.
        
        Args:
            X: Input features [N, D]
            y: Targets [N] or [N, 1]
            
        Returns:
            weights: [D, 1]
            bias: scalar
        """
        if y.ndim == 1:
            y = y.unsqueeze(1)
        
        if self.verbose:
            print(f"⚡ Fast Damping Initialization")
            print(f"   Features: {X.shape[1]} | Samples: {X.shape[0]}")
        
        # 1. Standardize for correlation computation
        X_mean = X.mean(dim=0, keepdim=True)
        X_std = X.std(dim=0, keepdim=True) + self.eps
        X_norm = (X - X_mean) / X_std
        
        y_mean = y.mean()
        y_std = y.std() + self.eps
        y_norm = (y - y_mean) / y_std
        
        # 2. Compute correlations with target
        numerator = torch.matmul(X_norm.T, y_norm) / X.shape[0]
        
        # 3. Compute feature correlation matrix
        corr_matrix = torch.matmul(X_norm.T, X_norm) / X.shape[0]
        
        # 4. Damping from neighbor correlations
        mask = ~torch.eye(corr_matrix.shape[0], dtype=torch.bool, device=X.device)
        neighbor_correlations = corr_matrix * mask
        damping = 1.0 + torch.sum(torch.abs(neighbor_correlations), dim=1, keepdim=True)
        
        # 5. Damped weights in standardized space
        weights_norm = numerator / damping
        
        # 6. Transform back to original scale
        weights = weights_norm * (y_std / X_std.T)
        bias = y_mean - torch.matmul(X_mean, weights).squeeze()
        
        if self.verbose:
            print(f"✅ Initialization complete")
        
        return weights, bias
    
    def initialize_layer(
        self,
        layer: nn.Linear,
        X_data: torch.Tensor,
        y_data: torch.Tensor
    ) -> None:
        """Initialize a PyTorch Linear layer in-place."""
        assert isinstance(layer, nn.Linear), "Layer must be nn.Linear"
        
        weights, bias = self.compute_weights(X_data, y_data)
        
        with torch.no_grad():
            layer.weight.data = weights.T
            if layer.bias is not None:
                layer.bias.data = bias if bias.ndim == 0 else bias.view(-1)


# Convenience functions
def scbi_init(
    X_data: torch.Tensor,
    y_data: torch.Tensor,
    n_samples: int = 10,
    sample_ratio: float = 0.5,
    ridge_alpha: float = 1.0,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function for SCBI initialization.
    
    Args:
        X_data: Input features [N, D]
        y_data: Targets [N] or [N, Output_Dim]
        n_samples: Number of stochastic subsets
        sample_ratio: Fraction of data per subset
        ridge_alpha: Regularization strength
        verbose: Print progress information
    
    Returns:
        weights: [D, Output_Dim]
        bias: [Output_Dim]
    
    Example:
        >>> weights, bias = scbi_init(X_train, y_train, n_samples=20)
    """
    initializer = SCBIInitializer(n_samples, sample_ratio, ridge_alpha, verbose)
    return initializer.compute_weights(X_data, y_data)


def fast_damping_init(
    X_data: torch.Tensor,
    y_data: torch.Tensor,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function for fast damping initialization.
    
    Args:
        X_data: Input features [N, D]
        y_data: Targets [N] or [N, 1]
        verbose: Print progress information
    
    Returns:
        weights: [D, 1]
        bias: scalar
    
    Example:
        >>> weights, bias = fast_damping_init(X_train, y_train)
    """
    initializer = FastDampingInitializer(verbose=verbose)
    return initializer.compute_weights(X_data, y_data)
