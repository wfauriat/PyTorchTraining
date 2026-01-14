import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Pyro imports
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

# Clear Pyro's parameter store
pyro.clear_param_store()

print("Loading and preprocessing data...")
# Load data
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split and normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test_scaled)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Define Bayesian Neural Network
class BayesianNN(PyroModule):
    def __init__(self, input_size=8, hidden_size=64, output_size=1):
        super().__init__()
        
        # First layer with priors on weights and biases
        self.fc1 = PyroModule[nn.Linear](input_size, hidden_size)
        self.fc1.weight = PyroSample(
            dist.Normal(0., 1.).expand([hidden_size, input_size]).to_event(2)
        )
        self.fc1.bias = PyroSample(
            dist.Normal(0., 1.).expand([hidden_size]).to_event(1)
        )
        
        # Second layer
        self.fc2 = PyroModule[nn.Linear](hidden_size, hidden_size // 2)
        self.fc2.weight = PyroSample(
            dist.Normal(0., 1.).expand([hidden_size // 2, hidden_size]).to_event(2)
        )
        self.fc2.bias = PyroSample(
            dist.Normal(0., 1.).expand([hidden_size // 2]).to_event(1)
        )
        
        # Output layer
        self.fc3 = PyroModule[nn.Linear](hidden_size // 2, output_size)
        self.fc3.weight = PyroSample(
            dist.Normal(0., 1.).expand([output_size, hidden_size // 2]).to_event(2)
        )
        self.fc3.bias = PyroSample(
            dist.Normal(0., 1.).expand([output_size]).to_event(1)
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x, y=None):
        # Forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc3(x).squeeze()
        
        # Observation noise (aleatoric uncertainty)
        sigma = pyro.sample("sigma", dist.Gamma(2.0, 1.0))
        
        # Likelihood
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        
        return mu

print("\nInitializing Bayesian Neural Network...")
model = BayesianNN(input_size=8, hidden_size=64, output_size=1)

# Create variational guide (approximate posterior)
guide = AutoDiagonalNormal(model)

# Setup optimizer and inference algorithm
adam = Adam({"lr": 0.01})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

print("\nTraining Bayesian Neural Network with Variational Inference...")
print("=" * 60)

# Training loop
num_iterations = 2000
losses = []

for step in range(num_iterations):
    loss = svi.step(X_train, y_train_tensor)
    losses.append(loss)
    
    if (step + 1) % 200 == 0:
        print(f"Iteration {step + 1}/{num_iterations}, Loss: {loss:.4f}")

print("\nTraining complete!")

#%%

# Make predictions with uncertainty
print("\nGenerating predictions with uncertainty...")

# Sample from posterior
num_samples = 100
predictive = Predictive(model, guide=guide, num_samples=num_samples)

# Get predictions on test set
with torch.no_grad():
    samples = predictive(X_test)
    predictions_scaled = samples['obs'].numpy()  # Shape: [num_samples, n_test]

print(f"Generated {num_samples} posterior samples")

# Transform predictions back to original scale
predictions_original = np.zeros_like(predictions_scaled)
for i in range(num_samples):
    predictions_original[i] = scaler_y.inverse_transform(
        predictions_scaled[i].reshape(-1, 1)
    ).flatten()

# Calculate statistics
mean_pred = predictions_original.mean(axis=0)
std_pred = predictions_original.std(axis=0)
lower_bound = np.percentile(predictions_original, 2.5, axis=0)  # 95% CI
upper_bound = np.percentile(predictions_original, 97.5, axis=0)

# Calculate metrics
mse = np.mean((mean_pred - y_test) ** 2)
mae = np.mean(np.abs(mean_pred - y_test))
r2 = 1 - np.sum((y_test - mean_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

print(f'\n{"="*60}')
print("Bayesian Neural Network Performance:")
print(f'{"="*60}')
print(f'Test MSE: {mse:.4f}')
print(f'Test MAE: {mae:.4f}')
print(f'Test R²:  {r2:.4f}')

# Calculate coverage (what % of true values fall within 95% CI)
coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
print(f'95% CI Coverage: {coverage*100:.2f}% (should be ~95%)')

# Visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training loss
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(losses, 'b-', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('ELBO Loss')
ax1.set_title('Variational Inference Training Loss')
ax1.grid(True, alpha=0.3)

# Plot 2: Posterior samples
ax2 = fig.add_subplot(gs[1, 0])
# sample_indices = np.argsort(y_test)[:100]
sample_indices = np.random.randint(0, y_test.shape[0], 100)
for i in range(min(20, num_samples)):  # Show 20 samples
    ax2.scatter(y_test[sample_indices], predictions_original[i][sample_indices],
               alpha=0.2, s=10, c='blue')
ax2.scatter(y_test[sample_indices], mean_pred[sample_indices],
           c='red', s=30, marker='x', linewidth=2, label='Posterior Mean', zorder=10)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'k--', linewidth=2, alpha=0.5)
ax2.set_xlabel('True Price ($100k)')
ax2.set_ylabel('Predicted Price ($100k)')
ax2.set_title('Posterior Samples (100 samples shown)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Predictions with credible intervals
ax3 = fig.add_subplot(gs[1, 1])

random_indices = np.random.choice(len(y_test), size=50, replace=False)
indices = random_indices[np.argsort(y_test[random_indices])]

ax3.fill_between(y_test[indices], lower_bound[indices], upper_bound[indices],
                 alpha=0.3, color='blue', label='95% Credible Interval')
ax3.scatter(y_test[indices], mean_pred[indices], c='red', s=30, zorder=10, label='Posterior Mean')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'k--', linewidth=2, alpha=0.5, label='Perfect prediction')
ax3.set_xlabel('True Price ($100k)')
ax3.set_ylabel('Predicted Price ($100k)')
ax3.set_title('Predictions with 95% Credible Intervals (50 random samples)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Prediction error vs uncertainty
ax4 = fig.add_subplot(gs[1, 2])
errors = np.abs(mean_pred - y_test)
ax4.scatter(std_pred, errors, alpha=0.5, s=20)
ax4.set_xlabel('Epistemic Uncertainty (Posterior Std)')
ax4.set_ylabel('Absolute Error')
ax4.set_title('Prediction Error vs Uncertainty')
correlation = np.corrcoef(std_pred, errors)[0, 1]
ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
        transform=ax4.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax4.grid(True, alpha=0.3)

# Plot 5: Predictions colored by uncertainty
ax5 = fig.add_subplot(gs[2, 0])
scatter = ax5.scatter(y_test, mean_pred, c=std_pred,
                     cmap='viridis', alpha=0.6, s=20)
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2)
ax5.set_xlabel('True Price ($100k)')
ax5.set_ylabel('Predicted Price ($100k)')
ax5.set_title('Predictions (color = uncertainty)')
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Uncertainty')
ax5.grid(True, alpha=0.3)

# Plot 6: Uncertainty distribution
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(std_pred, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax6.axvline(std_pred.mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {std_pred.mean():.3f}')
ax6.set_xlabel('Epistemic Uncertainty (Posterior Std)')
ax6.set_ylabel('Count')
ax6.set_title('Uncertainty Distribution')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Calibration plot
ax7 = fig.add_subplot(gs[2, 2])
# Check calibration at different confidence levels
confidence_levels = np.arange(0.1, 1.0, 0.1)
empirical_coverage = []
for conf in confidence_levels:
    alpha = (1 - conf) / 2
    lower = np.percentile(predictions_original, alpha * 100, axis=0)
    upper = np.percentile(predictions_original, (1 - alpha) * 100, axis=0)
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    empirical_coverage.append(coverage)

ax7.plot(confidence_levels, empirical_coverage, 'bo-', linewidth=2, markersize=8, label='Empirical')
ax7.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
ax7.set_xlabel('Expected Coverage')
ax7.set_ylabel('Empirical Coverage')
ax7.set_title('Calibration Plot')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.set_xlim([0, 1])
ax7.set_ylim([0, 1])

plt.suptitle('Bayesian Neural Network with Variational Inference',
            fontsize=16, fontweight='bold', y=0.995)
plt.show()

# Print high uncertainty predictions
print(f'\n{"="*80}')
print("High Uncertainty Predictions (top 10):")
print("="*80)
print(f"{'Index':>5} | {'True':>8} | {'Predicted':>10} | {'Error':>8} | {'Uncertainty':>11} | {'95% CI':>20}")
print("-"*80)
high_uncertainty_idx = np.argsort(std_pred)[-10:][::-1]
for idx in high_uncertainty_idx:
    error = mean_pred[idx] - y_test[idx]
    ci_width = upper_bound[idx] - lower_bound[idx]
    print(f"{idx:>5} | ${y_test[idx]:>7.2f} | ${mean_pred[idx]:>9.2f} | "
          f"${error:>7.2f} | {std_pred[idx]:>11.3f} | "
          f"[${lower_bound[idx]:.2f}, ${upper_bound[idx]:.2f}]")

# Calibration check
median_uncertainty = np.median(std_pred)
low_unc_mask = std_pred < median_uncertainty
high_unc_mask = std_pred >= median_uncertainty

low_unc_mae = np.mean(np.abs(mean_pred[low_unc_mask] - y_test[low_unc_mask]))
high_unc_mae = np.mean(np.abs(mean_pred[high_unc_mask] - y_test[high_unc_mask]))

print(f'\n{"="*80}')
print("Calibration Check:")
print("="*80)
print(f"Low uncertainty samples MAE:  ${low_unc_mae:.4f}")
print(f"High uncertainty samples MAE: ${high_unc_mae:.4f}")
print(f"Ratio (should be > 1 if well-calibrated): {high_unc_mae / low_unc_mae:.2f}")

# Extract learned observation noise (aleatoric uncertainty)
print(f'\n{"="*80}')
print("Uncertainty Decomposition:")
print("="*80)
sigma_samples = samples['sigma'].numpy()
print(f"Aleatoric uncertainty (observation noise σ):")
print(f"  Mean: {sigma_samples.mean():.4f}")
print(f"  Std:  {sigma_samples.std():.4f}")
print(f"Epistemic uncertainty (model uncertainty):")
print(f"  Mean: {std_pred.mean():.4f}")
print(f"  Std:  {std_pred.std():.4f}")

# Save some sample predictions to show posterior distribution
print(f'\n{"="*80}')
print("Sample Posterior Distributions (first 5 test samples):")
print("="*80)
for i in range(5):
    sample_preds = predictions_original[:, i]
    print(f"\nSample {i+1}:")
    print(f"  True value:      ${y_test[i]:.2f}")
    print(f"  Posterior mean:  ${mean_pred[i]:.2f}")
    print(f"  Posterior std:   ${std_pred[i]:.2f}")
    print(f"  95% CI:          [${lower_bound[i]:.2f}, ${upper_bound[i]:.2f}]")
    print(f"  Min/Max samples: [${sample_preds.min():.2f}, ${sample_preds.max():.2f}]")

print("\n" + "="*80)
print("Analysis complete!")