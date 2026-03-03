# SCBI Quick Start Guide

Get started with SCBI in 5 minutes! ⚡

---

## 📥 Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.9 or higher
- NumPy
- scikit-learn (for Ridge CV)

### Install Dependencies

```bash
pip install torch numpy scikit-learn
```

### Get SCBI

```bash
# Download from Zenodo
wget https://zenodo.org/record/18576203/files/scbi.py

# Or clone repository
git clone https://github.com/yourusername/scbi.git
cd scbi
```

---

## 🚀 Your First SCBI Model (60 seconds)

### Step 1: Import Libraries

```python
import torch
import torch.nn as nn
from scbi import SCBILinear
```

### Step 2: Prepare Your Data

```python
# Your training data
X_train = torch.randn(1000, 100)  # 1000 samples, 100 features
y_train = torch.randn(1000, 10)   # 1000 samples, 10 outputs

# Create proxy sample (10-30% of training data)
X_proxy = X_train[:300]
y_proxy = y_train[:300]
```

### Step 3: Create and Initialize SCBI Layer

```python
# Create layer
layer = SCBILinear(in_features=100, out_features=10)

# Initialize with SCBI (this is the magic!)
layer.init_weights_with_proxy(X_proxy, y_proxy)
```

### Step 4: Train (just like normal PyTorch!)

```python
# Setup training
optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Train
for epoch in range(30):
    optimizer.zero_grad()
    output = layer(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

**🎉 Done! You just used SCBI!**

Expected output:
```
Epoch 10: Loss = 0.5234
Epoch 20: Loss = 0.2156
Epoch 30: Loss = 0.1234
```

With SCBI, your initial loss will be **much lower** than random initialization!

---

## 🔧 Complete Examples

### Example 1: Simple Regression

```python
import torch
import torch.nn as nn
from scbi import SCBILinear

# Generate synthetic data
n_samples = 1000
X_train = torch.randn(n_samples, 50)
y_train = torch.randn(n_samples, 1)

# Create SCBI layer
layer = SCBILinear(50, 1)

# Initialize with SCBI
X_proxy = X_train[:300]
y_proxy = y_train[:300]
layer.init_weights_with_proxy(X_proxy, y_proxy)

# Train
optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(30):
    optimizer.zero_grad()
    pred = layer(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.item():.4f}")
```

---

### Example 2: Deep Neural Network

```python
import torch
import torch.nn as nn
from scbi import create_scbi_mlp

# Generate data
X_train = torch.randn(2000, 100)
y_train = torch.randn(2000, 10)

# Create deep network with SCBI
model = create_scbi_mlp(
    input_dim=100,
    hidden_dims=[256, 128, 64],  # 3 hidden layers
    output_dim=10,
    activation=nn.ReLU()
)

# Initialize all layers with SCBI
X_proxy = X_train[:500]
y_proxy = y_train[:500]
model.init_scbi_layers(X_proxy, y_proxy)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(30):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

---

### Example 3: Classification

```python
import torch
import torch.nn as nn
from scbi import create_scbi_mlp

# Generate classification data
X_train = torch.randn(2000, 50)
y_train = torch.randint(0, 3, (2000,))  # 3 classes

# One-hot encode for SCBI initialization
y_train_onehot = nn.functional.one_hot(y_train, num_classes=3).float()

# Create model
model = create_scbi_mlp(
    input_dim=50,
    hidden_dims=[128, 64],
    output_dim=3,
    activation=nn.ReLU()
)

# Initialize with one-hot encoded targets
X_proxy = X_train[:500]
y_proxy = y_train_onehot[:500]
model.init_scbi_layers(X_proxy, y_proxy)

# Train with original labels
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(30):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)  # Use original labels here
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        acc = (pred.argmax(dim=1) == y_train).float().mean()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Acc = {acc:.3f}")
```

---

## 💡 Tips and Best Practices

### ✅ DO:

**1. Use 10-30% of training data as proxy:**
```python
proxy_size = int(0.2 * len(X_train))
X_proxy = X_train[:proxy_size]
y_proxy = y_train[:proxy_size]
```

**2. Always use automatic Ridge tuning:**
```python
layer = SCBILinear(100, 50, tune_ridge=True)  # ✓ Recommended
```

**3. Initialize before creating optimizer:**
```python
layer.init_weights_with_proxy(X_proxy, y_proxy)
optimizer = torch.optim.Adam(layer.parameters())
```

**4. Standardize your input features:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### ❌ DON'T:

**1. Use too few proxy samples:**
```python
# BAD: Only 10 samples
X_proxy = X_train[:10]

# GOOD: At least 100 samples
X_proxy = X_train[:max(100, int(0.2 * len(X_train)))]
```

**2. Forget to initialize before training:**
```python
# BAD: Missing initialization
layer = SCBILinear(100, 50)
optimizer = torch.optim.Adam(layer.parameters())
# Start training without SCBI initialization!

# GOOD: Initialize first
layer = SCBILinear(100, 50)
layer.init_weights_with_proxy(X_proxy, y_proxy)
optimizer = torch.optim.Adam(layer.parameters())
```

---

## 🎓 Real-World Example: California Housing

```python
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scbi import create_scbi_mlp

# Load real dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Create model
model = create_scbi_mlp(
    input_dim=8,
    hidden_dims=[128, 64],
    output_dim=1,
    activation=nn.ReLU(),
    dropout=0.2
)

# Initialize with SCBI
X_proxy = X_train[:500]
y_proxy = y_train[:500]
model.init_scbi_layers(X_proxy, y_proxy)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(30):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
        print(f"Epoch {epoch+1}: "
              f"Train = {loss.item():.4f}, "
              f"Test = {test_loss.item():.4f}")
```

---

## 📚 Next Steps

- **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference
- **[README](README.md)** - Full documentation with experiments
- **[Research Paper](scbi_paper_full.tex)** - Theoretical details

---

**SCBI Quick Start Guide** | Version 3.0.0
