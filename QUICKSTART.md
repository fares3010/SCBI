# SCBI Quick Start Guide

## 🚀 Get Started in 2 Minutes

### Installation

No installation needed! Just download `scbi.py` and import it.

### Basic Usage

```python
from scbi import scbi_init
import torch

# Your data
X_train = torch.randn(1000, 50)  # [samples, features]
y_train = torch.randn(1000)      # [samples]

# Initialize weights
weights, bias = scbi_init(X_train, y_train)

# Use in your model
import torch.nn as nn
model = nn.Linear(50, 1)
with torch.no_grad():
    model.weight.data = weights.T  # Note the transpose!
    model.bias.data = bias
```

That's it! Your model now starts with data-informed weights instead of random initialization.

## 📊 Expected Performance

Based on our experiments:

| Task | Standard Init | SCBI Init | Improvement |
|------|--------------|-----------|-------------|
| **Regression** | 26,000 MSE | 300 MSE | **87× better** |
| **Classification** | 1.18 loss | 0.79 loss | **33% better** |

## 🎛️ Hyperparameters

### Default Settings (Work Well for Most Cases)

```python
weights, bias = scbi_init(
    X_train, 
    y_train,
    n_samples=10,      # Number of random subsets to average
    sample_ratio=0.5,  # Use 50% of data per subset
    ridge_alpha=1.0    # Regularization strength
)
```

### When to Adjust

**n_samples:**
- Increase (15-20) if you have lots of data and want stability
- Decrease (5) if you want faster initialization

**sample_ratio:**
- Increase (0.7-0.8) if you have clean, consistent data
- Keep low (0.3-0.5) if you have noisy or outlier-prone data

**ridge_alpha:**
- Increase (2.0-5.0) if your features are highly correlated
- Decrease (0.5) if your features are already well-conditioned

## 🔍 Classification Example

For multi-class classification, convert targets to one-hot encoding:

```python
import torch.nn.functional as F

# Your class labels (0, 1, 2, ...)
y_labels = torch.tensor([0, 2, 1, 0, ...])

# Convert to one-hot
y_onehot = F.one_hot(y_labels, num_classes=3).float()

# Initialize
from scbi import SCBIInitializer
initializer = SCBIInitializer(n_samples=10)
model = nn.Linear(50, 3)
initializer.initialize_layer(model, X_train, y_onehot)
```

## ⚡ High-Dimensional Data (D > 10,000)

Use the fast approximation:

```python
from scbi import fast_damping_init

X_train = torch.randn(1000, 20000)  # 20,000 features!
y_train = torch.randn(1000)

# Fast method - avoids expensive matrix inversion
weights, bias = fast_damping_init(X_train, y_train)
```

## 🧪 Complete Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from scbi import scbi_init

# 1. Prepare your data
X_train = torch.randn(1000, 50)
y_train = torch.randn(1000, 1)

# 2. Create model
model = nn.Linear(50, 1)

# 3. Initialize with SCBI
weights, bias = scbi_init(X_train, y_train, verbose=True)
with torch.no_grad():
    model.weight.data = weights.T
    model.bias.data = bias.squeeze()

# 4. Train normally
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## ⚠️ Common Mistakes

### Mistake 1: Forgetting to Transpose

```python
# ❌ WRONG
model.weight.data = weights  # Shape mismatch!

# ✅ CORRECT
model.weight.data = weights.T  # PyTorch stores weights as [out, in]
```

### Mistake 2: Using Class Indices Instead of One-Hot

```python
# ❌ WRONG (for SCBI initialization)
y_train = torch.tensor([0, 1, 2, 0, 1])  # Class indices

# ✅ CORRECT
y_train = F.one_hot(torch.tensor([0, 1, 2, 0, 1]), num_classes=3).float()
```

### Mistake 3: Not Standardizing Features

```python
from sklearn.preprocessing import StandardScaler

# ✅ GOOD PRACTICE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_train = torch.tensor(X_train, dtype=torch.float32)
```

## 🎯 Best Practices

1. **Always standardize your features** before using SCBI
2. **Use one-hot encoding** for classification targets
3. **Start with defaults** (n_samples=10, ridge_alpha=1.0)
4. **Profile on a subset** if you have very large datasets
5. **Compare with baselines** to measure actual improvement

## 📚 Next Steps

- See `example_usage.py` for complete experiments
- Read `README.md` for detailed documentation
- Check `PAPER_OUTLINE.md` for theoretical background

## 🆘 Troubleshooting

**"Matrix is singular" warning?**
→ Increase `ridge_alpha` (try 2.0 or 5.0)

**SCBI not helping?**
→ Your problem might be highly non-linear. SCBI works best for linear/weakly non-linear tasks.

**Out of memory?**
→ Use `fast_damping_init()` or reduce `n_samples`

**Getting nan/inf values?**
→ Check that your data is standardized and doesn't contain extreme values

## 💡 Pro Tips

- SCBI works best on the **first layer** of deep networks
- The improvement is most visible in the **first few epochs**
- For production, you can **cache** SCBI weights and reuse them
- SCBI is deterministic (given same random seed), so results are reproducible

---

**Happy Training! 🎉**

Questions? Open an issue on GitHub or contact Fares Ashraf.
