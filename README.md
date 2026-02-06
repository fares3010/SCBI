# SCBI: Stochastic Covariance-Based Initialization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Official implementation of the paper: "Stochastic Covariance-Based Initialization (SCBI): A Scalable Warm-Start Strategy for High-Dimensional Linear Models"**

> **TL;DR:** SCBI is a GPU-accelerated initialization method that calculates the optimal starting weights for linear layers using covariance statistics. It effectively "solves" linear regression tasks before the first epoch of training, replacing random initialization with a statistically grounded "warm start."

---

## 🚀 Key Results

SCBI significantly outperforms standard Random Initialization (He/Xavier) by approximating the closed-form solution via stochastic bagging.

| Dataset | Task | Improvement (Epoch 0) |
| :--- | :--- | :--- |
| **Synthetic High-Dim** | Binary Classification | **61.6%** reduction in loss |
| **California Housing** | Regression | **90.8%** reduction in MSE |
| **Forest Cover Type** | Multi-Class Classification | **26.2%** reduction in loss |

<p align="center">
  <!-- Replace this URL with your actual image path after uploading to GitHub, e.g., "assets/convergence_graph.png" -->
  <img src="https://via.placeholder.com/800x400?text=Insert+Your+Graph+Here" alt="Convergence Graph">
  <br>
  <em>Figure 1: Convergence comparison on Synthetic High-Dimensional Data.</em>
</p>

---

## 🛠️ Installation

Dependencies are minimal. You only need PyTorch and standard data libraries.

```bash
pip install torch numpy scikit-learn matplotlib
```
## ⚡ Usage

SCBI is designed to be a "drop-in" replacement for standard initialization. Instead of initializing with `nn.init.kaiming_normal_`, you calculate weights using `compute_scbi_weights` and assign them.

### 1. For Regression (Continuous Target)

```python
import torch
import torch.nn as nn
from scbi import compute_scbi_weights # Assuming you save the algorithm in scbi.py

# 1. Define your model
model = nn.Linear(input_dim, 1)

# 2. Calculate SCBI Weights (GPU Accelerated)
# Note: X_train and y_train must be Tensors
print("Calculating Warm Start...")
w_init, b_init = compute_scbi_weights(X_train, y_train, n_samples=10, ridge_alpha=1.0)

# 3. Assign Weights to Model
with torch.no_grad():
    model.weight.data = w_init.T
    model.bias.data = b_init

# 4. Train as normal (Adam/SGD)...
# You will see the loss start near 0.
```

### 2. For Classification (Multi-Class)

For classification, ensure your target `y` is One-Hot Encoded before passing it to SCBI.

```python
from scbi import compute_scbi_classification

# ... Load Data ...

# Calculate Weights for 7 Classes
w_init, b_init = compute_scbi_classification(X_train, y_one_hot, n_samples=10)

with torch.no_grad():
    model.weight.data = w_init.T
    model.bias.data = b_init.squeeze()
```

## 📖 Methodology
Standard initialization strategies (Xavier, He) are **semantically blind**—they initialize weights based on architecture dimensions, ignoring data statistics.

SCBI leverages the **Normal Equation** approximation:

$$\theta_{SCBI} = \frac{1}{K} \sum_{k=1}^{K} \left[ \left( \tilde{X}_k^T \tilde{X}_k + \lambda I \right)^{-1} \tilde{X}_k^T Y_k \right]$$

By computing this on random subsets (Bagging) using GPU matrix operations, we obtain a robust estimator of the global minimum without the $O(N^3)$ cost of full matrix inversion.

## 📄 Citation

If you use SCBI in your research, please cite our paper:

```bibtex
@article{scbi2026,
  title={Stochastic Covariance-Based Initialization (SCBI): A Scalable Warm-Start Strategy for High-Dimensional Linear Models},
  author={Fares Ashraf},
  journal={arXiv preprint},
  year={2026}
}
```
