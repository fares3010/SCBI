"""
SCBI Comprehensive Benchmark Suite
Publication-Quality Experiments and Visualizations

This script provides a complete proof-of-concept for SCBI with:
- Multiple real-world datasets
- Statistical significance testing
- Publication-ready figures
- Ablation studies
- Reproducible results

Author: Fares Ashraf
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import (
    fetch_california_housing,
    load_diabetes,
    load_breast_cancer,
    make_regression,
    make_classification
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats as scipy_stats
from time import time
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Import SCBI (make sure scbi.py is in the same directory)
from scbi import SCBILinear, create_scbi_mlp


# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class BenchmarkExperiment:
    """
    Comprehensive benchmarking framework for SCBI.
    """

    def __init__(self, n_runs=5, n_epochs=30, device='cpu'):
        self.n_runs = n_runs
        self.n_epochs = n_epochs
        self.device = device
        self.results = []

    def load_dataset(self, dataset_name):
        """Load and preprocess datasets."""
        print(f"\n{'='*70}")
        print(f"Loading Dataset: {dataset_name}")
        print('='*70)

        if dataset_name == 'california_housing':
            data = fetch_california_housing()
            X, y = data.data, data.target
            task = 'regression'

        elif dataset_name == 'diabetes':
            data = load_diabetes()
            X, y = data.data, data.target
            task = 'regression'

        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            X, y = data.data, data.target.astype(np.float32)
            task = 'classification'

        elif dataset_name == 'synthetic_highdim':
            X, y = make_regression(
                n_samples=2000,
                n_features=500,
                n_informative=250,
                noise=20.0,
                random_state=RANDOM_SEED
            )
            task = 'regression'

        elif dataset_name == 'synthetic_classification':
            X, y = make_classification(
                n_samples=2000,
                n_features=50,
                n_informative=30,
                n_classes=3,
                random_state=RANDOM_SEED
            )
            y = y.astype(np.float32)
            task = 'classification'

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device)

        if task == 'classification':
            # One-hot encode for SCBI initialization
            n_classes = int(y_train.max().item()) + 1
            y_train_onehot = torch.nn.functional.one_hot(
                y_train.long(), num_classes=n_classes
            ).float()
        else:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            y_train_onehot = y_train

        print(f"Task: {task}")
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        if task == 'classification':
            print(f"Classes: {n_classes}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_onehot': y_train_onehot,
            'task': task,
            'name': dataset_name
        }

    def train_model(self, model, X_train, y_train, X_test, y_test,
                    task, lr=0.01, verbose=False):
        """Train a model and track metrics."""

        # Loss function
        if task == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        test_losses = []
        train_times = []

        for epoch in range(self.n_epochs):
            start_time = time()

            # Training
            model.train()
            optimizer.zero_grad()

            if task == 'regression':
                pred = model(X_train)
                loss = criterion(pred, y_train)
            else:
                pred = model(X_train)
                loss = criterion(pred, y_train.long())

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_times.append(time() - start_time)

            # Testing
            model.eval()
            with torch.no_grad():
                if task == 'regression':
                    test_pred = model(X_test)
                    test_loss = criterion(test_pred, y_test)
                else:
                    test_pred = model(X_test)
                    test_loss = criterion(test_pred, y_test.long())

                test_losses.append(test_loss.item())

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}: "
                      f"Train Loss={loss.item():.4f}, "
                      f"Test Loss={test_loss.item():.4f}")

        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_times': train_times,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'initial_train_loss': train_losses[0],
            'initial_test_loss': test_losses[0]
        }

    def run_experiment(self, dataset_name):
        """Run complete experiment on one dataset."""

        data = self.load_dataset(dataset_name)

        # Prepare proxy sample
        proxy_size = min(500, int(0.3 * len(data['X_train'])))
        X_proxy = data['X_train'][:proxy_size]
        y_proxy = data['y_train_onehot'][:proxy_size]

        n_features = data['X_train'].shape[1]
        if data['task'] == 'regression':
            n_outputs = 1
        else:
            n_outputs = int(data['y_train'].max().item()) + 1

        results_runs = {'standard': [], 'scbi': []}

        print(f"\nRunning {self.n_runs} independent trials...")

        for run in range(self.n_runs):
            print(f"\nRun {run+1}/{self.n_runs}")
            print("-" * 50)

            # Standard Initialization
            print("Training with Standard Initialization...")
            model_std = nn.Linear(n_features, n_outputs).to(self.device)

            results_std = self.train_model(
                model_std,
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                data['task'],
                verbose=(run == 0)
            )
            results_runs['standard'].append(results_std)

            # SCBI Initialization
            print("\nTraining with SCBI Initialization...")
            model_scbi = SCBILinear(
                n_features, n_outputs,
                tune_ridge=True,
                n_samples=10
            ).to(self.device)

            # Initialize with SCBI
            init_start = time()
            model_scbi.init_weights_with_proxy(
                X_proxy, y_proxy,
                verbose=(run == 0)
            )
            init_time = time() - init_start

            results_scbi = self.train_model(
                model_scbi,
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                data['task'],
                verbose=(run == 0)
            )
            results_scbi['init_time'] = init_time
            results_runs['scbi'].append(results_scbi)

        # Aggregate statistics
        stats = self.compute_statistics(results_runs, data)

        # Store results
        self.results.append({
            'dataset': dataset_name,
            'task': data['task'],
            'stats': stats,
            'runs': results_runs
        })

        # Print summary
        self.print_summary(stats, dataset_name)

        return stats

    def compute_statistics(self, results_runs, data):
        """Compute statistical metrics across runs."""

        stats = {}

        for method in ['standard', 'scbi']:
            runs = results_runs[method]

            initial_train = [r['initial_train_loss'] for r in runs]
            initial_test = [r['initial_test_loss'] for r in runs]
            final_train = [r['final_train_loss'] for r in runs]
            final_test = [r['final_test_loss'] for r in runs]

            stats[method] = {
                'initial_train_mean': np.mean(initial_train),
                'initial_train_std': np.std(initial_train),
                'initial_test_mean': np.mean(initial_test),
                'initial_test_std': np.std(initial_test),
                'final_train_mean': np.mean(final_train),
                'final_train_std': np.std(final_train),
                'final_test_mean': np.mean(final_test),
                'final_test_std': np.std(final_test),
                'train_losses_all': [r['train_losses'] for r in runs],
                'test_losses_all': [r['test_losses'] for r in runs]
            }

            if method == 'scbi':
                init_times = [r['init_time'] for r in runs]
                stats[method]['init_time_mean'] = np.mean(init_times)
                stats[method]['init_time_std'] = np.std(init_times)

        # Compute improvements
        initial_improvement = (
            (stats['standard']['initial_train_mean'] -
             stats['scbi']['initial_train_mean']) /
            stats['standard']['initial_train_mean'] * 100
        )

        final_improvement = (
            (stats['standard']['final_test_mean'] -
             stats['scbi']['final_test_mean']) /
            stats['standard']['final_test_mean'] * 100
        )

        # Statistical significance test (paired t-test)
        initial_std = [r['initial_train_loss'] for r in results_runs['standard']]
        initial_scbi = [r['initial_train_loss'] for r in results_runs['scbi']]
        t_stat, p_value = scipy_stats.ttest_rel(initial_std, initial_scbi)

        stats['improvement'] = {
            'initial_pct': initial_improvement,
            'final_pct': final_improvement,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        return stats

    def print_summary(self, stats, dataset_name):
        """Print formatted summary of results."""

        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY: {dataset_name}")
        print('='*70)

        print("\nInitial Loss (Before Training):")
        print(f"  Standard:  {stats['standard']['initial_train_mean']:10.4f} "
              f"± {stats['standard']['initial_train_std']:.4f}")
        print(f"  SCBI:      {stats['scbi']['initial_train_mean']:10.4f} "
              f"± {stats['scbi']['initial_train_std']:.4f}")
        print(f"  Improvement: {stats['improvement']['initial_pct']:6.1f}%")

        if stats['improvement']['significant']:
            print(f"  ✅ Statistically significant (p={stats['improvement']['p_value']:.4f})")
        else:
            print(f"  ⚠️  Not significant (p={stats['improvement']['p_value']:.4f})")

        print(f"\nFinal Test Loss (After {self.n_epochs} Epochs):")
        print(f"  Standard:  {stats['standard']['final_test_mean']:10.4f} "
              f"± {stats['standard']['final_test_std']:.4f}")
        print(f"  SCBI:      {stats['scbi']['final_test_mean']:10.4f} "
              f"± {stats['scbi']['final_test_std']:.4f}")
        print(f"  Improvement: {stats['improvement']['final_pct']:6.1f}%")

        print(f"\nInitialization Time:")
        print(f"  SCBI:      {stats['scbi']['init_time_mean']:.3f}s "
              f"± {stats['scbi']['init_time_std']:.3f}s")

        print('='*70)

    def create_visualizations(self, output_dir='figures'):
        """Create publication-quality visualizations."""

        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("Creating Visualizations")
        print('='*70)

        # Figure 1: Training curves for all datasets
        self.plot_training_curves(output_dir)

        # Figure 2: Initial loss comparison
        self.plot_initial_loss_comparison(output_dir)

        # Figure 3: Convergence speedup
        self.plot_convergence_speedup(output_dir)

        # Figure 4: Statistical significance
        self.plot_statistical_significance(output_dir)

        # Figure 5: Ablation study
        self.plot_ablation_study(output_dir)

        print(f"\n✅ All figures saved to '{output_dir}/' directory")

    def plot_training_curves(self, output_dir):
        """Plot training curves with confidence intervals."""

        n_datasets = len(self.results)
        fig, axes = plt.subplots(2, (n_datasets + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()

        for idx, result in enumerate(self.results):
            ax = axes[idx]
            stats = result['stats']

            epochs = range(1, self.n_epochs + 1)

            # Standard init
            std_losses = np.array(stats['standard']['train_losses_all'])
            std_mean = std_losses.mean(axis=0)
            std_std = std_losses.std(axis=0)

            ax.plot(epochs, std_mean, 'b-', linewidth=2, label='Standard', alpha=0.8)
            ax.fill_between(
                epochs,
                std_mean - std_std,
                std_mean + std_std,
                color='b', alpha=0.2
            )

            # SCBI init
            scbi_losses = np.array(stats['scbi']['train_losses_all'])
            scbi_mean = scbi_losses.mean(axis=0)
            scbi_std = scbi_losses.std(axis=0)

            ax.plot(epochs, scbi_mean, 'r-', linewidth=2, label='SCBI', alpha=0.8)
            ax.fill_between(
                epochs,
                scbi_mean - scbi_std,
                scbi_mean + scbi_std,
                color='r', alpha=0.2
            )

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Training Loss')
            ax.set_title(f"{result['dataset'].replace('_', ' ').title()}")
            ax.legend(loc='best')
            ax.grid(alpha=0.3)

            # Add improvement text
            improvement = stats['improvement']['initial_pct']
            ax.text(
                0.95, 0.95,
                f"Initial: {improvement:+.1f}%",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        # Hide extra subplots
        for idx in range(n_datasets, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/fig1_training_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/fig1_training_curves.pdf', bbox_inches='tight')
        print("✓ Figure 1: Training curves")
        plt.close()

    def plot_initial_loss_comparison(self, output_dir):
        """Bar plot comparing initial losses."""

        fig, ax = plt.subplots(figsize=(10, 6))

        dataset_names = [r['dataset'].replace('_', ' ').title()
                        for r in self.results]
        x = np.arange(len(dataset_names))
        width = 0.35

        std_initial = [r['stats']['standard']['initial_train_mean']
                      for r in self.results]
        std_err = [r['stats']['standard']['initial_train_std']
                  for r in self.results]

        scbi_initial = [r['stats']['scbi']['initial_train_mean']
                       for r in self.results]
        scbi_err = [r['stats']['scbi']['initial_train_std']
                   for r in self.results]

        bars1 = ax.bar(x - width/2, std_initial, width,
                      yerr=std_err, label='Standard',
                      color='steelblue', alpha=0.8, capsize=5)
        bars2 = ax.bar(x + width/2, scbi_initial, width,
                      yerr=scbi_err, label='SCBI',
                      color='coral', alpha=0.8, capsize=5)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('Initial Training Loss')
        ax.set_title('Initial Loss Comparison: SCBI vs Standard Initialization')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add improvement percentages
        for i, result in enumerate(self.results):
            improvement = result['stats']['improvement']['initial_pct']
            y_pos = max(std_initial[i], scbi_initial[i]) * 1.1
            ax.text(i, y_pos, f"{improvement:+.1f}%",
                   ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/fig2_initial_loss.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/fig2_initial_loss.pdf', bbox_inches='tight')
        print("✓ Figure 2: Initial loss comparison")
        plt.close()

    def plot_convergence_speedup(self, output_dir):
        """Plot convergence speedup (epochs to target loss)."""

        fig, axes = plt.subplots(1, len(self.results), figsize=(15, 4))

        if len(self.results) == 1:
            axes = [axes]

        for idx, result in enumerate(self.results):
            ax = axes[idx]
            stats = result['stats']

            # Compute epochs to reach 2× initial SCBI loss
            target = stats['scbi']['initial_train_mean'] * 2

            std_losses = np.array(stats['standard']['train_losses_all'])
            scbi_losses = np.array(stats['scbi']['train_losses_all'])

            std_epochs = []
            scbi_epochs = []

            for run_std, run_scbi in zip(std_losses, scbi_losses):
                # Find first epoch below target
                std_idx = np.where(run_std < target)[0]
                scbi_idx = np.where(run_scbi < target)[0]

                std_epochs.append(std_idx[0] + 1 if len(std_idx) > 0 else self.n_epochs)
                scbi_epochs.append(scbi_idx[0] + 1 if len(scbi_idx) > 0 else 1)

            speedup = np.array(std_epochs) / np.array(scbi_epochs)

            # Box plot
            data = [std_epochs, scbi_epochs]
            bp = ax.boxplot(data, labels=['Standard', 'SCBI'],
                           patch_artist=True)

            bp['boxes'][0].set_facecolor('steelblue')
            bp['boxes'][1].set_facecolor('coral')

            ax.set_ylabel('Epochs to Target Loss')
            ax.set_title(f"{result['dataset'].replace('_', ' ').title()}\n"
                        f"Speedup: {speedup.mean():.1f}×")
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/fig3_convergence_speedup.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/fig3_convergence_speedup.pdf', bbox_inches='tight')
        print("✓ Figure 3: Convergence speedup")
        plt.close()

    def plot_statistical_significance(self, output_dir):
        """Plot p-values and effect sizes."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        dataset_names = [r['dataset'].replace('_', ' ').title()
                        for r in self.results]

        # P-values
        p_values = [r['stats']['improvement']['p_value'] for r in self.results]
        colors = ['green' if p < 0.05 else 'orange' for p in p_values]

        ax1.barh(dataset_names, p_values, color=colors, alpha=0.7)
        ax1.axvline(x=0.05, color='r', linestyle='--', linewidth=2, label='α=0.05')
        ax1.set_xlabel('P-value')
        ax1.set_title('Statistical Significance (Paired t-test)')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)

        # Effect sizes (improvement percentages)
        improvements = [r['stats']['improvement']['initial_pct'] for r in self.results]
        colors = ['green' if imp > 10 else 'orange' if imp > 5 else 'red'
                 for imp in improvements]

        ax2.barh(dataset_names, improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Improvement (%)')
        ax2.set_title('Initial Loss Improvement')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/fig4_significance.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/fig4_significance.pdf', bbox_inches='tight')
        print("✓ Figure 4: Statistical significance")
        plt.close()

    def plot_ablation_study(self, output_dir):
        """Ablation study: effect of hyperparameters."""

        # Use first dataset for ablation
        data = self.load_dataset(self.results[0]['dataset'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Ablation 1: Number of subsets (n_samples)
        n_samples_range = [1, 5, 10, 15, 20]
        self.run_ablation(data, axes[0], 'n_samples', n_samples_range)

        # Ablation 2: Sample ratio
        sample_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.run_ablation(data, axes[1], 'sample_ratio', sample_ratios)

        # Ablation 3: Ridge alpha (with tune_ridge=False)
        ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        self.run_ablation(data, axes[2], 'ridge_alpha', ridge_alphas, tune_ridge=False)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/fig5_ablation.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/fig5_ablation.pdf', bbox_inches='tight')
        print("✓ Figure 5: Ablation study")
        plt.close()

    def run_ablation(self, data, ax, param_name, param_values, **fixed_params):
        """Run ablation study for one parameter."""

        print(f"\nAblation: {param_name}")

        proxy_size = min(500, int(0.3 * len(data['X_train'])))
        X_proxy = data['X_train'][:proxy_size]
        y_proxy = data['y_train_onehot'][:proxy_size]

        n_features = data['X_train'].shape[1]
        n_outputs = y_proxy.shape[1] if y_proxy.ndim > 1 else 1

        initial_losses = []
        final_losses = []

        for value in param_values:
            print(f"  {param_name}={value}...", end=' ')

            # Create model with specific parameter
            kwargs = {'tune_ridge': True}
            kwargs.update(fixed_params)
            kwargs[param_name] = value

            model = SCBILinear(n_features, n_outputs, **kwargs).to(self.device)
            model.init_weights_with_proxy(X_proxy, y_proxy, verbose=False)

            # Quick training
            results = self.train_model(
                model,
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                data['task'],
                verbose=False
            )

            initial_losses.append(results['initial_train_loss'])
            final_losses.append(results['final_test_loss'])
            print(f"Initial={results['initial_train_loss']:.2f}")

        # Plot
        ax.plot(param_values, initial_losses, 'o-', linewidth=2,
               markersize=8, label='Initial Loss', color='coral')
        ax.plot(param_values, final_losses, 's-', linewidth=2,
               markersize=8, label='Final Loss', color='steelblue')
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('Loss')
        ax.set_title(f'Effect of {param_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(alpha=0.3)

        if param_name == 'ridge_alpha':
            ax.set_xscale('log')

    def export_results(self, output_dir='results'):
        """Export results to CSV and LaTeX tables."""

        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("Exporting Results")
        print('='*70)

        # Create results dataframe
        data_rows = []

        for result in self.results:
            stats = result['stats']
            row = {
                'Dataset': result['dataset'],
                'Task': result['task'],
                'Standard_Initial_Mean': stats['standard']['initial_train_mean'],
                'Standard_Initial_Std': stats['standard']['initial_train_std'],
                'SCBI_Initial_Mean': stats['scbi']['initial_train_mean'],
                'SCBI_Initial_Std': stats['scbi']['initial_train_std'],
                'Standard_Final_Mean': stats['standard']['final_test_mean'],
                'Standard_Final_Std': stats['standard']['final_test_std'],
                'SCBI_Final_Mean': stats['scbi']['final_test_mean'],
                'SCBI_Final_Std': stats['scbi']['final_test_std'],
                'Improvement_Pct': stats['improvement']['initial_pct'],
                'P_Value': stats['improvement']['p_value'],
                'Significant': stats['improvement']['significant'],
                'Init_Time_Mean': stats['scbi']['init_time_mean'],
                'Init_Time_Std': stats['scbi']['init_time_std']
            }
            data_rows.append(row)

        df = pd.DataFrame(data_rows)

        # Save CSV
        csv_path = f'{output_dir}/results.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV: {csv_path}")

        # Create LaTeX table
        latex_table = self.create_latex_table(df)
        latex_path = f'{output_dir}/results_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"✓ LaTeX: {latex_path}")

        # Save JSON
        json_path = f'{output_dir}/results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
        print(f"✓ JSON: {json_path}")

    def create_latex_table(self, df):
        """Create publication-ready LaTeX table."""

        latex = r"""\begin{table}[htbp]
\centering
\caption{SCBI Performance Across Datasets}
\label{tab:scbi_results}
\begin{tabular}{lcccc}
\toprule
Dataset & \multicolumn{2}{c}{Initial Loss} & Improvement & $p$-value \\
        & Standard & SCBI & (\%) & \\
\midrule
"""

        for _, row in df.iterrows():
            dataset = row['Dataset'].replace('_', ' ').title()
            std_init = f"{row['Standard_Initial_Mean']:.2f}$\pm${row['Standard_Initial_Std']:.2f}"
            scbi_init = f"{row['SCBI_Initial_Mean']:.2f}$\pm${row['SCBI_Initial_Std']:.2f}"
            improvement = f"{row['Improvement_Pct']:.1f}"
            p_val = f"{row['P_Value']:.4f}"

            sig = r"\textbf{*}" if row['Significant'] else ""

            latex += f"{dataset} & {std_init} & {scbi_init} & {improvement}{sig} & {p_val} \\\\\n"

        latex += r"""\bottomrule
\multicolumn{5}{l}{\small * Statistically significant at $\alpha=0.05$}
\end{tabular}
\end{table}"""

        return latex


def main():
    """Run complete benchmark suite."""

    print("\n" + "="*70)
    print("SCBI COMPREHENSIVE BENCHMARK SUITE")
    print("Publication-Quality Proof of Concept")
    print("="*70)
    print(f"\nAuthor: Fares Ashraf")
    print(f"DOI: 10.5281/zenodo.18576203")
    print(f"Random Seed: {RANDOM_SEED}")
    print("="*70)

    # Initialize benchmark
    benchmark = BenchmarkExperiment(
        n_runs=5,          # 5 independent runs for statistical significance
        n_epochs=30,       # 30 epochs per experiment
        device='cpu'       # Change to 'cuda' if GPU available
    )

    # Datasets to benchmark
    datasets = [
        'california_housing',
        'diabetes',
        'breast_cancer',
        'synthetic_highdim',
        'synthetic_classification'
    ]

    # Run experiments
    for dataset in datasets:
        benchmark.run_experiment(dataset)

    # Create visualizations
    benchmark.create_visualizations(output_dir='figures')

    # Export results
    benchmark.export_results(output_dir='results')

    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  figures/fig1_training_curves.png (PDF)")
    print("  figures/fig2_initial_loss.png (PDF)")
    print("  figures/fig3_convergence_speedup.png (PDF)")
    print("  figures/fig4_significance.png (PDF)")
    print("  figures/fig5_ablation.png (PDF)")
    print("  results/results.csv")
    print("  results/results_table.tex")
    print("  results/results.json")
    print("\n✨ Ready for publication and repository!")
    print("="*70)


if __name__ == "__main__":
    main()
