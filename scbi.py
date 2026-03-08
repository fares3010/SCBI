"""
SCBI: Stochastic Covariance-Based Initialization
A Scalable Warm-Start Strategy for High-Dimensional Linear Models.

Author: Fares Ashraf
Version: 1.2.0 (Production Release)
DOI: 10.5281/zenodo.18850507
License: MIT

This module provides a GPU-accelerated initialization strategy that
approximates the closed-form solution of linear/dense layers using
stochastic bagging, dynamic Ridge CV, and memory-efficient mean-centering.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

__version__ = "3.0.0"
__author__ = "Fares Ashraf"
__doi__ = "10.5281/zenodo.18576203"


class SCBILinear(nn.Module):
    """
    SCBI Linear Layer with Dynamic Ridge CV and Memory Optimizations.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: If True, adds learnable bias (default: True)
        n_samples: Number of stochastic subsets for bagging (default: 10)
        sample_ratio: Fraction of proxy data per subset (default: 0.5)
        ridge_alpha: Base Ridge regularization strength (default: 1.0)
        tune_ridge: If True, dynamically tunes ridge_alpha via nested CV (default: True)
        cv_folds: Number of folds for Ridge CV (default: 5)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_samples: int = 10,
        sample_ratio: float = 0.5,
        ridge_alpha: float = 1.0,
        tune_ridge: bool = True,
        cv_folds: int = 5
    ):
        super(SCBILinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # SCBI hyperparameters
        self.n_samples = n_samples
        self.sample_ratio = sample_ratio
        self.ridge_alpha = ridge_alpha
        self.tune_ridge = tune_ridge
        self.cv_folds = cv_folds

        # Core parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self._scbi_initialized = False
        self._reset_parameters()

    def _reset_parameters(self):
        """Standard Kaiming initialization fallback before SCBI is applied."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _tune_ridge_cv(self, X: torch.Tensor, y: torch.Tensor, lambdas: list) -> float:
        """
        Performs K-Fold Cross Validation strictly within the proxy batch
        to dynamically find the mathematically optimal Ridge penalty.
        """
        n_samples = X.shape[0]
        if n_samples < self.cv_folds:
            return self.ridge_alpha # Fallback if proxy batch is extremely small

        indices = torch.randperm(n_samples, device=X.device)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        fold_size = n_samples // self.cv_folds
        best_lambda = None
        best_loss = float('inf')

        identity = torch.eye(self.in_features, device=X.device, dtype=X.dtype)

        for lam in lambdas:
            val_losses = []

            for k in range(self.cv_folds):
                val_start = k * fold_size
                val_end = (k + 1) * fold_size if k < self.cv_folds - 1 else n_samples

                # Create masks for O(1) memory slicing
                val_idx = torch.arange(val_start, val_end, device=X.device)
                train_idx = torch.cat([
                    torch.arange(0, val_start, device=X.device),
                    torch.arange(val_end, n_samples, device=X.device)
                ])

                X_train, y_train = X_shuffled[train_idx], y_shuffled[train_idx]
                X_val, y_val = X_shuffled[val_idx], y_shuffled[val_idx]

                # Memory-Efficient Mean Centering
                X_mean, y_mean = X_train.mean(dim=0), y_train.mean(dim=0)
                X_c = X_train - X_mean
                y_c = y_train - y_mean

                XTX = torch.matmul(X_c.T, X_c)
                XTy = torch.matmul(X_c.T, y_c)

                try:
                    w = torch.linalg.solve(XTX + lam * identity, XTy)
                except RuntimeError:
                    w = torch.matmul(torch.linalg.pinv(XTX + lam * identity), XTy)

                # Reconstruct bias
                b = y_mean - torch.matmul(w.T, X_mean)
                b = b.squeeze()

                # Validate
                preds = torch.matmul(X_val, w) + b
                mse = torch.nn.functional.mse_loss(preds, y_val)
                val_losses.append(mse.item())

            avg_loss = sum(val_losses) / len(val_losses)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_lambda = lam

        return best_lambda

    def _tune_classification_cv(
            self, X: torch.Tensor, y_raw: torch.Tensor, lambdas: list, scales: list, num_classes: int
        ) -> Tuple[float, float]:
            """
            2D Grid Search CV for Classification.
            Tunes both Ridge penalty (λ) and Confidence Scale (S) by evaluating
            the linear approximations against the true Cross-Entropy / BCE loss.
            """
            n_samples = X.shape[0]
            if n_samples < self.cv_folds:
                return self.ridge_alpha, scales[0]

            indices = torch.randperm(n_samples, device=X.device)
            X_shuffled = X[indices]
            y_shuffled = y_raw[indices]

            fold_size = n_samples // self.cv_folds
            best_lambda = lambdas[0]
            best_scale = scales[0]
            best_loss = float('inf')

            identity = torch.eye(self.in_features, device=X.device, dtype=X.dtype)
            is_binary = (self.out_features == 1)
            criterion = torch.nn.BCEWithLogitsLoss() if is_binary else torch.nn.CrossEntropyLoss()

            for scale in scales:
                # 1. Map labels to logits for the current scale
                if is_binary:
                    y_mapped = (y_shuffled.float().view(-1, 1) * 2.0 - 1.0) * scale
                else:
                    y_one_hot = torch.nn.functional.one_hot(y_shuffled.long(), num_classes=num_classes).float()
                    y_mapped = (y_one_hot * 2.0 - 1.0) * scale

                for lam in lambdas:
                    val_losses = []

                    for k in range(self.cv_folds):
                        val_start = k * fold_size
                        val_end = (k + 1) * fold_size if k < self.cv_folds - 1 else n_samples

                        val_idx = torch.arange(val_start, val_end, device=X.device)
                        train_idx = torch.cat([
                            torch.arange(0, val_start, device=X.device),
                            torch.arange(val_end, n_samples, device=X.device)
                        ])

                        X_train, y_train_map = X_shuffled[train_idx], y_mapped[train_idx]
                        X_val, y_val_raw = X_shuffled[val_idx], y_shuffled[val_idx]

                        # 2. Solve Ridge Regression
                        X_mean, y_mean = X_train.mean(dim=0), y_train_map.mean(dim=0)
                        X_c, y_c = X_train - X_mean, y_train_map - y_mean

                        XTX = torch.matmul(X_c.T, X_c)
                        XTy = torch.matmul(X_c.T, y_c)

                        try:
                            w = torch.linalg.solve(XTX + lam * identity, XTy)
                        except RuntimeError:
                            w = torch.matmul(torch.linalg.pinv(XTX + lam * identity), XTy)

                        b = y_mean - torch.matmul(w.T, X_mean)
                        b = b.squeeze()

                        # 3. Evaluate using the TRUE Classification Loss Function
                        preds = torch.matmul(X_val, w) + b

                        if is_binary:
                            loss = criterion(preds.view(-1), y_val_raw.float().view(-1))
                        else:
                            loss = criterion(preds, y_val_raw.long())

                        val_losses.append(loss.item())

                    avg_loss = sum(val_losses) / len(val_losses)

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_lambda = lam
                        best_scale = scale

            return best_lambda, best_scale

    def init_weights_with_proxy(
        self,
        proxy_x: torch.Tensor,
        proxy_y: Optional[torch.Tensor] = None,
        task: str = "regression",
        confidence_scale: float = 6.0,  # Acts as fallback if tuning is disabled
        verbose: bool = True
    ) -> None:
        """
        Main initialization function. Computes proxy statistics via bagging,
        tunes Ridge penalty (and scale for classification), and assigns optimal weights.
        """
        with torch.no_grad():
            n_total = proxy_x.shape[0]
            n_features = self.in_features
            n_outputs = self.out_features
            subset_size = max(1, int(n_total * self.sample_ratio))

            active_lambda = self.ridge_alpha
            active_scale = confidence_scale

            # --- CLASSIFICATION LOGIC & 2D GRID SEARCH ---
            if proxy_y is not None and task == "classification":
                is_binary = (self.out_features == 1)

                # Standardize to raw labels
                if proxy_y.ndim == 1 or proxy_y.shape[1] == 1:
                    proxy_y_raw = proxy_y.squeeze()
                    num_classes = 2 if is_binary else int(proxy_y_raw.max().item()) + 1
                else:
                    proxy_y_raw = torch.argmax(proxy_y, dim=1)
                    num_classes = proxy_y.shape[1]

                # 2D Grid Search (Ridge λ + Confidence Scale)
                if self.tune_ridge:
                    lambdas_space = [0.01, 0.1, 1.0, 10.0, 100.0]
                    scales_space = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                    active_lambda, active_scale = self._tune_classification_cv(
                        proxy_x, proxy_y_raw, lambdas_space, scales_space, num_classes
                    )

                if verbose:
                    print(f"   ⚙️ Mapped logit space (±{active_scale}) | 🔍 CV Ridge (λ): {active_lambda}")

                # Map targets using the mathematically optimal scale
                if is_binary:
                    proxy_y = (proxy_y_raw.float().view(-1, 1) * 2.0 - 1.0) * active_scale
                else:
                    y_one_hot = torch.nn.functional.one_hot(proxy_y_raw.long(), num_classes=num_classes).float()
                    proxy_y = (y_one_hot * 2.0 - 1.0) * active_scale

            # --- REGRESSION LOGIC (Standard CV) ---
            else:
                if proxy_y is None:
                    proxy_y = torch.matmul(proxy_x, self.weight.T)
                if proxy_y.ndim == 1:
                    proxy_y = proxy_y.unsqueeze(1)

                if self.tune_ridge:
                    search_space = [0.01, 0.1, 1.0, 10.0, 100.0]
                    active_lambda = self._tune_ridge_cv(proxy_x, proxy_y, search_space)
                    if verbose:
                        print(f"   🔍 CV Optimal Ridge (λ): {active_lambda}")

            if verbose:
                print(f"🚀 SCBI Initialization")
                print(f"   Proxy: {n_total} samples | Layer: [{n_features} → {n_outputs}]")

            # ... (Rest of the Bagging Loop remains exactly the same!) ...
            accumulated_weights = torch.zeros((n_features, n_outputs), device=proxy_x.device, dtype=proxy_x.dtype)
            accumulated_bias = torch.zeros((1, n_outputs), device=proxy_x.device, dtype=proxy_x.dtype)

            identity = torch.eye(n_features, device=proxy_x.device, dtype=proxy_x.dtype)

            # --- STOCHASTIC BAGGING LOOP ---
            for i in range(self.n_samples):
                indices = torch.randperm(n_total, device=proxy_x.device)[:subset_size]
                X_sub = proxy_x[indices]
                y_sub = proxy_y[indices]

                # Memory Efficient Mean Centering
                X_mean, y_mean = X_sub.mean(dim=0), y_sub.mean(dim=0)
                X_c, y_c = X_sub - X_mean, y_sub - y_mean

                # Solve Normal Equation: (X^T X + λI)^-1 X^T Y
                XTX = torch.matmul(X_c.T, X_c)
                XTy = torch.matmul(X_c.T, y_c)

                try:
                    w_solved = torch.linalg.solve(XTX + active_lambda * identity, XTy)
                except RuntimeError:
                    # Fallback for highly collinear or rank-deficient batches
                    w_solved = torch.matmul(torch.linalg.pinv(XTX + active_lambda * identity), XTy)

                # Reconstruct bias: b = y_mean - w^T * X_mean
                b_solved = y_mean - torch.matmul(w_solved.T, X_mean)

                accumulated_weights += w_solved
                accumulated_bias += b_solved.unsqueeze(0)

            # Average the bagged models and assign to layer parameters
            self.weight.data = (accumulated_weights / self.n_samples).T
            if self.bias is not None:
                bias_data = accumulated_bias / self.n_samples
                self.bias.data = bias_data.squeeze() if bias_data.ndim == 2 else bias_data

            if verbose:
                print(f"   ✅ Complete | Weight std: {self.weight.std():.4f}\n")

        self._scbi_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard linear forward pass."""
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'scbi_initialized={self._scbi_initialized}'
        )


class SCBISequential(nn.Sequential):
    """
    Sequential container that automatically propagates proxy data
    through hidden layers to initialize an entire deep network.
    """
    def init_scbi_layers(
        self,
        proxy_x: torch.Tensor,
        proxy_y: Optional[torch.Tensor] = None,
        task: str = "regression",         
        confidence_scale: float = 2.0,    
        verbose: bool = True
    ) -> None:
        if verbose:
            print("="*60)
            print("SCBI Sequential Network Initialization")
            print("="*60)

        current_activation = proxy_x

        for i, module in enumerate(self):
            if isinstance(module, SCBILinear):
                # Only pass the true target 'y' to the very last SCBI layer
                is_last_scbi = not any(isinstance(m, SCBILinear) for m in list(self)[i+1:])
                target = proxy_y if (is_last_scbi and proxy_y is not None) else None

                # Only apply the classification mapping to the final layer
                layer_task = task if (is_last_scbi and target is not None) else "regression"

                module.init_weights_with_proxy(
                    current_activation,
                    target,
                    task=layer_task,                      
                    confidence_scale=confidence_scale,    
                    verbose=verbose
                )

                with torch.no_grad():
                    current_activation = module(current_activation)
            else:
                # Pass through non-SCBI layers (e.g., ReLUs, Dropouts)
                with torch.no_grad():
                    current_activation = module(current_activation)

        if verbose:
            print("="*60)
            print("✅ All SCBI layers initialized successfully!")
            print("="*60 + "\n")


def create_scbi_mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    activation: nn.Module = None,
    dropout: float = 0.0,
    **scbi_kwargs
) -> SCBISequential:
    """Utility factory to build an MLP fully equipped with SCBILinear layers."""
    if activation is None:
        activation = nn.ReLU()

    layers = []

    # Input layer
    layers.append(SCBILinear(input_dim, hidden_dims[0], **scbi_kwargs))
    layers.append(activation)
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    # Hidden layers
    for i in range(len(hidden_dims) - 1):
        layers.append(SCBILinear(hidden_dims[i], hidden_dims[i+1], **scbi_kwargs))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Output layer
    layers.append(SCBILinear(hidden_dims[-1], output_dim, **scbi_kwargs))

    return SCBISequential(*layers)


def scbi_init(
    X_data: torch.Tensor,
    y_data: torch.Tensor,
    n_samples: int = 10,
    sample_ratio: float = 0.5,
    ridge_alpha: float = 1.0,
    tune_ridge: bool = True,
    task: str = "regression",         
    confidence_scale: float = 2.0,    
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to extract SCBI optimal weights and biases
    directly without instantiating a PyTorch module.
    """
    layer = SCBILinear(
        in_features=X_data.shape[1],
        out_features=1 if y_data.ndim == 1 else y_data.shape[1], # Will be overridden if one-hot mapping triggers
        n_samples=n_samples,
        sample_ratio=sample_ratio,
        ridge_alpha=ridge_alpha,
        tune_ridge=tune_ridge
    )

    layer.init_weights_with_proxy(
        X_data,
        y_data,
        task=task,                            
        confidence_scale=confidence_scale,    
        verbose=verbose
    )
    return layer.weight.data.T, layer.bias.data

__all__ = ['SCBILinear', 'SCBISequential', 'create_scbi_mlp', 'scbi_init']
