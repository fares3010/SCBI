# SCBI API Documentation

Complete API reference for SCBI v3.0.0

---

## Table of Contents

1. [Core Classes](#core-classes)
   - [SCBILinear](#scbilinear)
   - [SCBISequential](#scbisequential)
2. [Helper Functions](#helper-functions)
   - [create_scbi_mlp](#create_scbi_mlp)
3. [Usage Examples](#usage-examples)
4. [Advanced Usage](#advanced-usage)
5. [Error Handling](#error-handling)

---

## Core Classes

### SCBILinear

A drop-in replacement for `nn.Linear` with SCBI initialization support.

```python
class SCBILinear(nn.Module):
    """
    Linear layer with Stochastic Covariance-Based Initialization.
    
    Inherits from nn.Linear and adds data-dependent initialization
    via ridge regression with stochastic bagging.
    """
```

#### Constructor

```python
SCBILinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    n_samples: int = 10,
    sample_ratio: float = 0.5,
    ridge_alpha: float = 1.0,
    tune_ridge: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | *required* | Size of input features |
| `out_features` | `int` | *required* | Size of output features |
| `bias` | `bool` | `True` | If `True`, adds learnable bias |
| `n_samples` | `int` | `10` | Number of bagging subsets for ensemble |
| `sample_ratio` | `float` | `0.5` | Fraction of proxy data per subset (0-1) |
| `ridge_alpha` | `float` | `1.0` | Ridge regularization strength (if `tune_ridge=False`) |
| `tune_ridge` | `bool` | `True` | If `True`, auto-tune ridge via cross-validation |
| `device` | `torch.device` | `None` | Device to place layer on |
| `dtype` | `torch.dtype` | `None` | Data type for parameters |

**Returns:**
- `SCBILinear` instance

**Example:**

```python
import torch
from scbi import SCBILinear

# Create layer
layer = SCBILinear(
    in_features=100,
    out_features=50,
    bias=True,
    n_samples=10,        # Default: good for most cases
    sample_ratio=0.5,    # Default: balanced bias-variance
    tune_ridge=True      # Recommended: auto-tune regularization
)

# Layer is ready to use (has random initialization by default)
output = layer(input_tensor)
```

---

#### Methods

##### `init_weights_with_proxy()`

Initialize layer weights using SCBI with proxy sample.

```python
init_weights_with_proxy(
    X_proxy: torch.Tensor,
    y_proxy: torch.Tensor,
    verbose: bool = True
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_proxy` | `torch.Tensor` | *required* | Input features, shape `(N, in_features)` |
| `y_proxy` | `torch.Tensor` | *required* | Target values, shape `(N, out_features)` or `(N,)` |
| `verbose` | `bool` | `True` | If `True`, print initialization details |

**Returns:**
- `None` (modifies layer weights in-place)

**Raises:**
- `ValueError` if `X_proxy` or `y_proxy` have wrong shapes
- `RuntimeError` if ridge regression fails

**Example:**

```python
# Prepare proxy sample (10-30% of training data)
X_proxy = X_train[:500]  # Shape: (500, 100)
y_proxy = y_train[:500]  # Shape: (500, 50) or (500,)

# Initialize with SCBI
layer.init_weights_with_proxy(X_proxy, y_proxy, verbose=True)

# Output:
# 🚀 SCBI Initialization
#    Proxy: 500 samples | Layer: [100 → 50]
#    🔍 CV Optimal Ridge (λ): 1.0
#    ✅ Complete | Weight std: 0.3421
```

---

##### `forward()`

Forward pass through the layer.

```python
forward(input: torch.Tensor) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `torch.Tensor` | Input tensor, shape `(*, in_features)` |

**Returns:**
- `torch.Tensor`: Output tensor, shape `(*, out_features)`

**Example:**

```python
# Single sample
x = torch.randn(100)          # Shape: (100,)
y = layer(x)                  # Shape: (50,)

# Batch
x = torch.randn(32, 100)      # Shape: (32, 100)
y = layer(x)                  # Shape: (32, 50)
```

---

##### `get_init_loss()`

Compute initial loss before any training (diagnostic method).

```python
get_init_loss(
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: Optional[nn.Module] = None
) -> float
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `torch.Tensor` | *required* | Input features |
| `y` | `torch.Tensor` | *required* | Target values |
| `criterion` | `nn.Module` | `nn.MSELoss()` | Loss function |

**Returns:**
- `float`: Initial loss value

**Example:**

```python
# Check initial loss after SCBI initialization
init_loss = layer.get_init_loss(X_train, y_train)
print(f"Initial loss: {init_loss:.4f}")
# Output: Initial loss: 2.3456
```

---

### SCBISequential

Sequential container for multiple SCBI layers with automatic layer-by-layer initialization.

```python
class SCBISequential(nn.Sequential):
    """
    Sequential model with SCBI layer support.
    
    Automatically initializes all SCBILinear layers in sequence
    by propagating proxy samples through initialized layers.
    """
```

#### Constructor

```python
SCBISequential(*args)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `*args` | `nn.Module` | Variable number of modules |

**Returns:**
- `SCBISequential` instance

**Example:**

```python
from scbi import SCBILinear, SCBISequential
import torch.nn as nn

# Create sequential model
model = SCBISequential(
    SCBILinear(100, 128),
    nn.ReLU(),
    SCBILinear(128, 64),
    nn.ReLU(),
    SCBILinear(64, 10)
)
```

---

#### Methods

##### `init_scbi_layers()`

Initialize all SCBI layers in the model sequentially.

```python
init_scbi_layers(
    X_proxy: torch.Tensor,
    y_proxy: torch.Tensor,
    verbose: bool = True
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_proxy` | `torch.Tensor` | *required* | Input features, shape `(N, in_features)` |
| `y_proxy` | `torch.Tensor` | *required* | Target values, shape `(N, out_features)` |
| `verbose` | `bool` | `True` | If `True`, print progress |

**Returns:**
- `None` (modifies model weights in-place)

**How it works:**
1. Initialize first layer with `(X_proxy, y_proxy)`
2. Propagate `X_proxy` through first layer to get hidden activations
3. Initialize second layer with `(hidden_activations, y_proxy)`
4. Repeat for all SCBI layers

**Example:**

```python
# Prepare proxy sample
X_proxy = X_train[:500]  # Shape: (500, 100)
y_proxy = y_train[:500]  # Shape: (500, 10)

# Initialize all layers
model.init_scbi_layers(X_proxy, y_proxy, verbose=True)

# Output:
# 🚀 SCBI Layer 1: [100 → 128]
#    ✅ Initialized | Loss: 12.34
# 🚀 SCBI Layer 2: [128 → 64]
#    ✅ Initialized | Loss: 8.76
# 🚀 SCBI Layer 3: [64 → 10]
#    ✅ Initialized | Loss: 2.45
```

---

## Helper Functions

### create_scbi_mlp

Create a multi-layer perceptron with SCBI initialization.

```python
create_scbi_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: nn.Module = nn.ReLU(),
    dropout: float = 0.0,
    batch_norm: bool = False,
    **scbi_kwargs
) -> SCBISequential
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | `int` | *required* | Input feature dimension |
| `hidden_dims` | `List[int]` | *required* | Hidden layer dimensions |
| `output_dim` | `int` | *required* | Output dimension |
| `activation` | `nn.Module` | `nn.ReLU()` | Activation function |
| `dropout` | `float` | `0.0` | Dropout probability (0-1) |
| `batch_norm` | `bool` | `False` | If `True`, add BatchNorm layers |
| `**scbi_kwargs` | `dict` | `{}` | Additional arguments for SCBILinear |

**Returns:**
- `SCBISequential`: Configured model

**Example:**

```python
from scbi import create_scbi_mlp
import torch.nn as nn

# Create deep network
model = create_scbi_mlp(
    input_dim=100,
    hidden_dims=[256, 128, 64],
    output_dim=10,
    activation=nn.ReLU(),
    dropout=0.2,
    batch_norm=True,
    n_samples=10,        # SCBI parameter
    tune_ridge=True      # SCBI parameter
)

# Initialize
X_proxy = X_train[:500]
y_proxy = y_train[:500]
model.init_scbi_layers(X_proxy, y_proxy)

# Train
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(epochs):
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
```

---

## Usage Examples

### Example 1: Single Layer Regression

```python
import torch
from scbi import SCBILinear

# Generate data
X_train = torch.randn(1000, 50)
y_train = torch.randn(1000, 10)

# Create and initialize layer
layer = SCBILinear(50, 10, tune_ridge=True)
layer.init_weights_with_proxy(X_train[:300], y_train[:300])

# Train
optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(30):
    optimizer.zero_grad()
    output = layer(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

---

### Example 2: Deep Network Classification

```python
import torch
import torch.nn as nn
from scbi import create_scbi_mlp

# Generate data
X_train = torch.randn(2000, 100)
y_train = torch.randint(0, 3, (2000,))  # 3 classes

# One-hot encode for SCBI initialization
y_train_onehot = nn.functional.one_hot(y_train, num_classes=3).float()

# Create model
model = create_scbi_mlp(
    input_dim=100,
    hidden_dims=[128, 64, 32],
    output_dim=3,
    activation=nn.ReLU(),
    batch_norm=True,
    dropout=0.2
)

# Initialize with SCBI
X_proxy = X_train[:500]
y_proxy = y_train_onehot[:500]
model.init_scbi_layers(X_proxy, y_proxy)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(30):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)  # Use original labels
    loss.backward()
    optimizer.step()
```

---

### Example 3: Custom Architecture

```python
import torch.nn as nn
from scbi import SCBILinear, SCBISequential

# Custom model with mixed SCBI and standard layers
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scbi_part = SCBISequential(
            SCBILinear(100, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            SCBILinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.standard_part = nn.Sequential(
            nn.Linear(128, 64),  # Standard layer
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.scbi_part(x)
        x = self.standard_part(x)
        return x
    
    def init_scbi(self, X_proxy, y_proxy):
        self.scbi_part.init_scbi_layers(X_proxy, y_proxy)

# Use the model
model = CustomModel()
model.init_scbi(X_train[:500], y_train[:500])
```

---

## Advanced Usage

### Manual Ridge Tuning

```python
# Disable automatic tuning and set ridge manually
layer = SCBILinear(
    100, 50,
    tune_ridge=False,
    ridge_alpha=10.0  # Manually set
)

layer.init_weights_with_proxy(X_proxy, y_proxy)
```

---

### Custom Proxy Size

```python
# Use larger proxy sample for better initialization
proxy_size = min(1000, int(0.3 * len(X_train)))
X_proxy = X_train[:proxy_size]
y_proxy = y_train[:proxy_size]

layer.init_weights_with_proxy(X_proxy, y_proxy)
```

---

### GPU Usage

```python
# Create layer on GPU
device = torch.device('cuda')
layer = SCBILinear(100, 50, device=device)

# Move data to GPU
X_proxy = X_train[:500].to(device)
y_proxy = y_train[:500].to(device)

# Initialize
layer.init_weights_with_proxy(X_proxy, y_proxy)

# Train on GPU
for epoch in range(epochs):
    output = layer(X_train.to(device))
    loss = criterion(output, y_train.to(device))
    ...
```

---

### Multi-Output Regression

```python
# Multiple output dimensions
layer = SCBILinear(100, 50)  # 100 inputs, 50 outputs

# Prepare multi-dimensional targets
X_proxy = X_train[:500]      # Shape: (500, 100)
y_proxy = y_train[:500]      # Shape: (500, 50)

layer.init_weights_with_proxy(X_proxy, y_proxy)
```

---

## Error Handling

### Common Errors and Solutions

#### 1. Shape Mismatch

```python
# ERROR
layer = SCBILinear(100, 50)
X_proxy = torch.randn(500, 200)  # Wrong input dim
layer.init_weights_with_proxy(X_proxy, y_proxy)

# SOLUTION
# Ensure X_proxy.shape[1] == layer.in_features
X_proxy = torch.randn(500, 100)  # Correct
```

---

#### 2. Target Shape Issues

```python
# ERROR - For single output regression
layer = SCBILinear(100, 1)
y_proxy = torch.randn(500)  # Shape: (500,) - missing dimension

# SOLUTION
y_proxy = torch.randn(500, 1)  # Shape: (500, 1) - correct
```

---

#### 3. Small Proxy Sample

```python
# WARNING: Too few samples
X_proxy = X_train[:10]  # Only 10 samples!
layer.init_weights_with_proxy(X_proxy, y_proxy)

# RECOMMENDATION
# Use at least 100 samples, ideally 10-30% of training data
proxy_size = max(100, int(0.2 * len(X_train)))
X_proxy = X_train[:proxy_size]
```

---

#### 4. Ridge Singularity

```python
# ERROR: Singular matrix (very rare)
# Usually happens with ridge_alpha too small

# SOLUTION: Use larger ridge_alpha or enable auto-tuning
layer = SCBILinear(100, 50, tune_ridge=True)  # Recommended
# OR
layer = SCBILinear(100, 50, ridge_alpha=1.0)  # Manual
```

---

## Best Practices

### 1. Proxy Sample Size

```python
# Good practice: 10-30% of training data, minimum 100 samples
proxy_size = max(100, min(500, int(0.2 * len(X_train))))
X_proxy = X_train[:proxy_size]
y_proxy = y_train[:proxy_size]
```

### 2. Always Use Auto Ridge Tuning

```python
# ✓ RECOMMENDED
layer = SCBILinear(100, 50, tune_ridge=True)

# ✗ NOT RECOMMENDED (unless you know optimal λ)
layer = SCBILinear(100, 50, tune_ridge=False, ridge_alpha=0.1)
```

### 3. Initialize Before Training

```python
# ✓ CORRECT ORDER
layer = SCBILinear(100, 50)
layer.init_weights_with_proxy(X_proxy, y_proxy)  # Initialize first
optimizer = torch.optim.Adam(layer.parameters())  # Then create optimizer

# ✗ WRONG ORDER (optimizer might not see initialized weights)
optimizer = torch.optim.Adam(layer.parameters())
layer.init_weights_with_proxy(X_proxy, y_proxy)
```

### 4. Check Initial Loss

```python
# Verify initialization improved initial loss
layer.init_weights_with_proxy(X_proxy, y_proxy)

init_loss = layer.get_init_loss(X_train, y_train)
print(f"Initial loss after SCBI: {init_loss:.4f}")

# Should be much lower than random initialization
```

---

## Version History

### v3.0.0 (Current)
- Added automatic Ridge CV tuning
- Improved numerical stability
- GPU support
- Memory-efficient implementation

### v2.0.0
- Added rearrangement correlation
- Unified API

### v1.0.0
- Initial release
- Basic SCBI implementation

---

## See Also

- [Quick Start Guide](QUICKSTART.md)
- [Complete README](README.md)
- [Research Paper](scbi_paper_full.tex)
- [Benchmark Scripts](benchmark_publication.py)

---

## Support

For issues or questions:
- **Email:** farsashraf44@gmail.com
- **DOI:** [10.5281/zenodo.18576203](https://doi.org/10.5281/zenodo.18576203)

---

**SCBI API Documentation v3.0.0** | Last updated: March 2026
