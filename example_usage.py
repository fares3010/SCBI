"""
Complete SCBI Example
Demonstrates regression and classification with performance comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
from scbi import SCBIInitializer
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import seaborn as sns
import time


def run_experiment(task_type="regression", visualize=True):
    """
    Run a complete SCBI experiment comparing standard vs SCBI initialization.
    
    Args:
        task_type: 'regression' or 'classification'
        visualize: Whether to plot results
    
    Returns:
        Dictionary with results and metrics
    """
    print(f"\n{'='*60}")
    print(f"SCBI EXPERIMENT: {task_type.upper()}")
    print(f"{'='*60}\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Generate Data
    if task_type == "regression":
        X, y = make_regression(
            n_samples=2000, 
            n_features=50, 
            n_informative=30,
            noise=10.0, 
            random_state=42
        )
        y = y.reshape(-1, 1)
        output_dim = 1
        criterion = nn.MSELoss()
        lr = 0.01
    else:  # classification
        X, y = make_classification(
            n_samples=2000, 
            n_features=50, 
            n_classes=3, 
            n_informative=30,
            n_clusters_per_class=1, 
            random_state=42
        )
        # Convert to One-Hot for SCBI
        y_one_hot = torch.nn.functional.one_hot(
            torch.tensor(y).long(), 
            num_classes=3
        ).float()
        output_dim = 3
        criterion = nn.CrossEntropyLoss()
        lr = 0.01

    # 2. Standardize Features (Critical for Neural Networks!)
    print("Preprocessing data...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. Convert to Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    if task_type == "regression":
        y_tensor = torch.tensor(y, dtype=torch.float32)
        y_target_for_loss = y_tensor
        y_scbi_target = y_tensor
    else:
        # For training loss, PyTorch CrossEntropy expects class indices
        y_tensor = torch.tensor(y, dtype=torch.long)
        y_target_for_loss = y_tensor
        # For SCBI, we need one-hot encoding
        y_scbi_target = y_one_hot

    # 4. Train/Test Split
    split_idx = 1600
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_test = y_target_for_loss[:split_idx], y_target_for_loss[split_idx:]
    y_train_scbi = y_scbi_target[:split_idx]

    # 5. Define Model Architecture
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.linear(x)

    # Create two identical models
    model_std = SimpleModel(50, output_dim)
    model_scbi = SimpleModel(50, output_dim)

    # 6. Initialize SCBI Model
    print("\nInitializing model with SCBI...")
    initializer = SCBIInitializer(
        n_samples=10, 
        sample_ratio=0.5, 
        ridge_alpha=1.0,
        verbose=True
    )
    initializer.initialize_layer(model_scbi.linear, X_train, y_train_scbi)

    # 7. Training Setup
    optimizer_std = optim.SGD(model_std.parameters(), lr=lr)
    optimizer_scbi = optim.SGD(model_scbi.parameters(), lr=lr)

    losses_std_train = []
    losses_scbi_train = []
    losses_std_test = []
    losses_scbi_test = []

    n_epochs = 5
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 60)

    # 8. Training Loop
    for epoch in range(n_epochs):
        # ===== Standard Model =====
        model_std.train()
        optimizer_std.zero_grad()
        pred_std = model_std(X_train)
        
        if task_type == "regression":
            loss_std = criterion(pred_std, y_train)
        else:
            loss_std = criterion(pred_std, y_train)
        
        loss_std.backward()
        optimizer_std.step()
        losses_std_train.append(loss_std.item())
        
        # Test loss
        model_std.eval()
        with torch.no_grad():
            pred_std_test = model_std(X_test)
            if task_type == "regression":
                test_loss_std = criterion(pred_std_test, y_test)
            else:
                test_loss_std = criterion(pred_std_test, y_test)
            losses_std_test.append(test_loss_std.item())

        # ===== SCBI Model =====
        model_scbi.train()
        optimizer_scbi.zero_grad()
        pred_scbi = model_scbi(X_train)
        
        if task_type == "regression":
            loss_scbi = criterion(pred_scbi, y_train)
        else:
            loss_scbi = criterion(pred_scbi, y_train)
        
        loss_scbi.backward()
        optimizer_scbi.step()
        losses_scbi_train.append(loss_scbi.item())
        
        # Test loss
        model_scbi.eval()
        with torch.no_grad():
            pred_scbi_test = model_scbi(X_test)
            if task_type == "regression":
                test_loss_scbi = criterion(pred_scbi_test, y_test)
            else:
                test_loss_scbi = criterion(pred_scbi_test, y_test)
            losses_scbi_test.append(test_loss_scbi.item())

        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs}:")
        print(f"  Standard Init - Train: {loss_std.item():.4f}, Test: {test_loss_std.item():.4f}")
        print(f"  SCBI Init     - Train: {loss_scbi.item():.4f}, Test: {test_loss_scbi.item():.4f}")
        improvement = ((loss_std.item() - loss_scbi.item()) / loss_std.item()) * 100
        print(f"  Improvement: {improvement:.1f}%\n")

    # 9. Results Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    initial_improvement = ((losses_std_train[0] - losses_scbi_train[0]) / losses_std_train[0]) * 100
    final_improvement = ((losses_std_train[-1] - losses_scbi_train[-1]) / losses_std_train[-1]) * 100
    
    print(f"\nInitial Loss (Epoch 1):")
    print(f"  Standard: {losses_std_train[0]:.4f}")
    print(f"  SCBI:     {losses_scbi_train[0]:.4f}")
    print(f"  Improvement: {initial_improvement:.1f}%")
    
    print(f"\nFinal Loss (Epoch {n_epochs}):")
    print(f"  Standard: {losses_std_train[-1]:.4f}")
    print(f"  SCBI:     {losses_scbi_train[-1]:.4f}")
    print(f"  Improvement: {final_improvement:.1f}%")
    
    print(f"\nTest Set Performance:")
    print(f"  Standard: {losses_std_test[-1]:.4f}")
    print(f"  SCBI:     {losses_scbi_test[-1]:.4f}")
    
    # 10. Visualization
    if visualize:
        plt.figure(figsize=(14, 5))
        
        # Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, n_epochs+1), losses_std_train, 
                 'b--', linewidth=2.5, label='Standard Init', alpha=0.7)
        plt.plot(range(1, n_epochs+1), losses_scbi_train, 
                 'r-', linewidth=2.5, label='SCBI Init')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title(f'{task_type.capitalize()} - Training Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        # Test Loss
        plt.subplot(1, 2, 2)
        plt.plot(range(1, n_epochs+1), losses_std_test, 
                 'b--', linewidth=2.5, label='Standard Init', alpha=0.7)
        plt.plot(range(1, n_epochs+1), losses_scbi_test, 
                 'r-', linewidth=2.5, label='SCBI Init')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Test Loss', fontsize=12)
        plt.title(f'{task_type.capitalize()} - Test Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        filename = f'scbi_{task_type}_results.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved as '{filename}'")
        plt.show()

    # 11. Return results
    return {
        'task_type': task_type,
        'train_losses_std': losses_std_train,
        'train_losses_scbi': losses_scbi_train,
        'test_losses_std': losses_std_test,
        'test_losses_scbi': losses_scbi_test,
        'initial_improvement_pct': initial_improvement,
        'final_improvement_pct': final_improvement,
    }


def run_hyperparameter_search():
    """
    Demonstrate hyperparameter tuning for SCBI.
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH")
    print("="*60 + "\n")
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=50, noise=10.0, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    # Split
    X_train, X_val = X_tensor[:800], X_tensor[800:]
    y_train, y_val = y_tensor[:800], y_tensor[800:]
    
    # Grid search
    param_grid = {
        'n_samples': [5, 10, 20],
        'ridge_alpha': [0.5, 1.0, 2.0],
    }
    
    best_loss = float('inf')
    best_params = None
    
    print("Testing parameter combinations...")
    for n_samples in param_grid['n_samples']:
        for ridge_alpha in param_grid['ridge_alpha']:
            initializer = SCBIInitializer(
                n_samples=n_samples,
                ridge_alpha=ridge_alpha,
                verbose=False
            )
            weights, bias = initializer.compute_weights(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = X_val @ weights + bias
            loss = torch.mean((y_pred - y_val) ** 2).item()
            
            print(f"  n_samples={n_samples:2d}, ridge_alpha={ridge_alpha:.1f} → MSE={loss:.2f}")
            
            if loss < best_loss:
                best_loss = loss
                best_params = (n_samples, ridge_alpha)
    
    print(f"\n✅ Best parameters:")
    print(f"   n_samples: {best_params[0]}")
    print(f"   ridge_alpha: {best_params[1]}")
    print(f"   Validation MSE: {best_loss:.2f}")


if __name__ == "__main__":
    # Run regression experiment
    results_reg = run_experiment("regression", visualize=True)
    
    # Run classification experiment
    results_cls = run_experiment("classification", visualize=True)
    
    # Run hyperparameter search
    run_hyperparameter_search()
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    print("\n📈 Summary:")
    print(f"  Regression  - Initial improvement: {results_reg['initial_improvement_pct']:.1f}%")
    print(f"  Regression  - Final improvement:   {results_reg['final_improvement_pct']:.1f}%")
    print(f"  Classification - Initial improvement: {results_cls['initial_improvement_pct']:.1f}%")
    print(f"  Classification - Final improvement:   {results_cls['final_improvement_pct']:.1f}%")


# ==========================================
# 1. DATASET PREPARATION HELPERS
# ==========================================

def get_california_housing():
    print("   Loading California Housing...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test, 1 # 1 output dim

def get_mnist():
    print("   Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Flatten for MLP
    X_train = train_dataset.data.view(-1, 28*28).float() / 255.0
    y_train = train_dataset.targets
    X_test = test_dataset.data.view(-1, 28*28).float() / 255.0
    y_test = test_dataset.targets

    return X_train, y_train, X_test, y_test, 10

def get_cifar10():
    print("   Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

    # Use a subset for speed in this demo (5000 samples)
    subset_indices = torch.randperm(len(train_dataset))[:5000]

    # Flatten images: [N, 3*32*32]
    X_train = torch.stack([train_dataset[i][0] for i in subset_indices]).view(5000, -1)
    y_train = torch.tensor([train_dataset[i][1] for i in subset_indices])

    test_subset = torch.randperm(len(test_dataset))[:1000]
    X_test = torch.stack([test_dataset[i][0] for i in test_subset]).view(1000, -1)
    y_test = torch.tensor([test_dataset[i][1] for i in test_subset])

    return X_train, y_train, X_test, y_test, 10

# ==========================================
# 2. UNIVERSAL EXPERIMENT RUNNER
# ==========================================

def run_comparison(task_name, X_train, y_train, X_test, y_test, output_dim, epochs=10):
    input_dim = X_train.shape[1]

    # --- Define Models ---
    # We use a simple MLP: Input -> Hidden -> Output
    # This tests if SCBI helps propagate signals through layers
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(input_dim, 128)
            self.act = nn.ReLU()
            self.out = nn.Linear(128, output_dim)

        def forward(self, x):
            x = self.act(self.hidden(x))
            return self.out(x)

    model_std = MLP()
    model_scbi = MLP()

    # --- Initialize SCBI ---
    # We initialize BOTH layers for maximum effect
    print(f"⚡ [SCBI] Initializing {task_name}...")
    start_time = time.time()

    # 1. Init Hidden Layer
    # We treat the hidden layer as a multi-target regression problem
    # We map Input -> Random Projection (to simulate targets)
    # This is a trick: 'Auto-Encoder' style init or just fit to first layer of Standard model
    # For simplicity here: We only init the OUTPUT layer (Linear Probing style)
    # OR we can do a greedy layer-wise init. Let's do the Output Layer Only for stability.

    # Init Output Layer
    # We need features at the hidden layer first!
    with torch.no_grad():
        hidden_feats = model_scbi.act(model_scbi.hidden(X_train[:2000])) # Get hidden activations

    if output_dim > 1:
        y_ohe = F.one_hot(y_train[:2000].long(), output_dim).float()
    else:
        y_ohe = y_train[:2000]

    initializer = SCBIInitializer(n_samples=10, sample_ratio=0.8, verbose=False)
    initializer.initialize_layer(model_scbi.out, hidden_feats, y_ohe)

    print(f"   Done in {time.time()-start_time:.2f}s")

    # --- Training Loop ---
    criterion = nn.MSELoss() if output_dim == 1 else nn.CrossEntropyLoss()
    opt_std = optim.SGD(model_std.parameters(), lr=0.01 if output_dim==1 else 0.1)
    opt_scbi = optim.SGD(model_scbi.parameters(), lr=0.01 if output_dim==1 else 0.1)

    history = {'std': [], 'scbi': []}

    # Create Batches
    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    print(f"   Training for {epochs} epochs...")
    for epoch in range(epochs):
        # Train Standard
        epoch_loss_std = 0
        for xb, yb in dl:
            opt_std.zero_grad()
            out = model_std(xb)
            loss = criterion(out, yb.view(-1, 1) if output_dim==1 else yb.long())
            loss.backward()
            opt_std.step()
            epoch_loss_std += loss.item()
        history['std'].append(epoch_loss_std / len(dl))

        # Train SCBI
        epoch_loss_scbi = 0
        for xb, yb in dl:
            opt_scbi.zero_grad()
            out = model_scbi(xb)
            loss = criterion(out, yb.view(-1, 1) if output_dim==1 else yb.long())
            loss.backward()
            opt_scbi.step()
            epoch_loss_scbi += loss.item()
        history['scbi'].append(epoch_loss_scbi / len(dl))

    return history

# ==========================================
# 3. RUN ALL AND PLOT
# ==========================================

print("=== STARTING EXPERIMENTS ===")

# Exp 1: Housing (Regression)
X_h_tr, y_h_tr, X_h_te, y_h_te, out_h = get_california_housing()
hist_housing = run_comparison("Housing", X_h_tr, y_h_tr, X_h_te, y_h_te, out_h, epochs=15)

# Exp 2: MNIST (Simple Class)
X_m_tr, y_m_tr, X_m_te, y_m_te, out_m = get_mnist()
hist_mnist = run_comparison("MNIST", X_m_tr, y_m_tr, X_m_te, y_m_te, out_m, epochs=10)

# Exp 3: CIFAR (Complex Class)
X_c_tr, y_c_tr, X_c_te, y_c_te, out_c = get_cifar10()
hist_cifar = run_comparison("CIFAR-10", X_c_tr, y_c_tr, X_c_te, y_c_te, out_c, epochs=15)

# --- CORRECTED PLOTTING BLOCK ---
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# We pack the data into tuples: (Title, DataDict, Y-Label)
datasets_to_plot = [
    ("California Housing (Regression)", hist_housing, "MSE Loss"),
    ("MNIST (Simple Classification)", hist_mnist, "CrossEntropy"),
    ("CIFAR-10 (Complex Classification)", hist_cifar, "CrossEntropy")
]

for ax, (title, data, ylabel) in zip(axes, datasets_to_plot):
    # keys in run_comparison were 'std' and 'scbi'
    epochs = range(len(data['std']))

    # 1. Plot Lines (Using correct keys 'std' and 'scbi')
    ax.plot(epochs, data['std'], 'b--', lw=2, label='Standard Init', alpha=0.6)
    ax.plot(epochs, data['scbi'], 'r-', lw=3, label='SCBI Init')

    # 2. Styling
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Epochs")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Annotation (Calculate % Improvement at Epoch 0)
    start_std = data['std'][0]
    start_scbi = data['scbi'][0]

    # Avoid division by zero if loss is 0 (unlikely but safe)
    if start_std > 1e-6:
        imp = (1 - start_scbi/start_std) * 100

        # Add arrow annotation
        ax.annotate(f'{imp:.0f}% Lower\nInitial Loss',
                    xy=(0, start_scbi), xytext=(len(epochs)//4, start_std),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10, color='darkred', fontweight='bold')

plt.tight_layout()
plt.savefig("scbi_comprehensive_benchmark.png", dpi=300)
print("\n✅ Experiments Complete. Plot saved to 'scbi_comprehensive_benchmark.png'")
plt.show()