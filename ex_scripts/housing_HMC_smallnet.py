import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# PyMC imports
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az

print("="*80)
print("HAMILTONIAN MONTE CARLO WITH PyMC")
print("="*80)

# Load and preprocess data
print("\nLoading and preprocessing data...")
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Use subset for computational feasibility
print("Using subset of data for HMC...")
X = X[:2000]
y = y[:2000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Convert to shared variables (allows updating for prediction)
X_train_shared = pytensor.shared(X_train_scaled)
y_train_shared = pytensor.shared(y_train_scaled)

# Network architecture
input_size = 8
hidden_size = 10  # Small network for HMC
output_size = 1

n_params = (input_size * hidden_size + hidden_size +  # Layer 1
            hidden_size * output_size + output_size +  # Layer 2
            1)  # sigma

print(f"\nNetwork architecture: {input_size} → {hidden_size} → {output_size}")
print(f"Total parameters: {n_params}")

# Build Bayesian Neural Network in PyMC
print("\nBuilding Bayesian Neural Network model...")

with pm.Model() as bnn_model:
    # Priors for layer 1 weights and biases
    weights_1 = pm.Normal('w1', mu=0, sigma=1, 
                          shape=(input_size, hidden_size))
    bias_1 = pm.Normal('b1', mu=0, sigma=1, 
                       shape=(hidden_size,))
    
    # Priors for layer 2 weights and biases
    weights_2 = pm.Normal('w2', mu=0, sigma=1, 
                          shape=(hidden_size, output_size))
    bias_2 = pm.Normal('b2', mu=0, sigma=1, 
                       shape=(output_size,))
    
    # Prior for observation noise
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Neural network forward pass
    # Layer 1: X @ W1 + b1, then ReLU
    hidden = pm.math.dot(X_train_shared, weights_1) + bias_1
    hidden_activated = pm.math.tanh(hidden)  # Use tanh (more stable than ReLU in PyMC)
    
    # Layer 2: hidden @ W2 + b2
    output = pm.math.dot(hidden_activated, weights_2) + bias_2
    mu = output.flatten()
    
    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train_shared)

print("Model built successfully!")
print("\nModel summary:")
print(bnn_model)

# Sample from posterior using NUTS (HMC variant)
print("\n" + "="*80)
print("Running NUTS sampler...")
print("="*80)

num_samples = 1000
num_tune = 500
num_chains = 2

print(f"\nSampler configuration:")
print(f"  Tuning (warmup) steps: {num_tune}")
print(f"  Samples per chain: {num_samples}")
print(f"  Number of chains: {num_chains}")
print(f"  Total samples: {num_samples * num_chains}")
print(f"\nEstimated time: 5-15 minutes (depends on your CPU)")

start_time = time.time()

with bnn_model:
    trace = pm.sample(
        draws=num_samples,
        tune=num_tune,
        chains=num_chains,
        cores=num_chains,  # Parallel chains
        target_accept=0.9,  # Higher = more accurate but slower
        return_inferencedata=True,
        progressbar=True
    )

elapsed = time.time() - start_time
print(f"\nSampling completed in {elapsed/60:.1f} minutes")

# Diagnostics
print("\n" + "="*80)
print("MCMC Diagnostics:")
print("="*80)

# Summary statistics
summary = az.summary(trace, hdi_prob=0.95)
print("\nPosterior Summary (first few parameters):")
print(summary.head(10))

# Check convergence (R-hat should be < 1.01)
print("\nConvergence diagnostics:")
rhat_values = summary['r_hat'].values
print(f"  R-hat range: [{rhat_values.min():.4f}, {rhat_values.max():.4f}]")
print(f"  All R-hat < 1.01: {np.all(rhat_values < 1.01)}")

# Effective sample size
ess = summary['ess_bulk'].values
print(f"  ESS range: [{ess.min():.0f}, {ess.max():.0f}]")
print(f"  Mean ESS: {ess.mean():.0f}")

# Make predictions on test set
print("\n" + "="*80)
print("Generating predictions...")
print("="*80)

# Update shared variables for prediction
X_train_shared.set_value(X_test_scaled)
y_train_shared.set_value(np.zeros(len(X_test_scaled)))  # Dummy values

with bnn_model:
    # Posterior predictive sampling
    ppc = pm.sample_posterior_predictive(
        trace,
        var_names=['y_obs'],
        progressbar=True
    )

# Extract predictions
predictions_scaled = ppc.posterior_predictive['y_obs'].values
# Shape: [chains, samples, n_test] -> flatten to [total_samples, n_test]
predictions_scaled = predictions_scaled.reshape(-1, len(X_test_scaled))

print(f"Generated {predictions_scaled.shape[0]} posterior predictive samples")

# Transform back to original scale
predictions_original = scaler_y.inverse_transform(predictions_scaled.T).T

# Calculate statistics
mean_pred = predictions_original.mean(axis=0)
std_pred = predictions_original.std(axis=0)
lower_bound = np.percentile(predictions_original, 2.5, axis=0)
upper_bound = np.percentile(predictions_original, 97.5, axis=0)

# Metrics
mse = np.mean((mean_pred - y_test) ** 2)
mae = np.mean(np.abs(mean_pred - y_test))
r2 = 1 - np.sum((y_test - mean_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))

print(f'\n{"="*80}')
print("PyMC HMC Performance:")
print(f'{"="*80}')
print(f'Test MSE: {mse:.4f}')
print(f'Test MAE: {mae:.4f}')
print(f'Test R²:  {r2:.4f}')
print(f'95% CI Coverage: {coverage*100:.2f}% (should be ~95%)')

# Visualizations
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Trace plots
ax1 = fig.add_subplot(gs[0, :])
az.plot_trace(trace, var_names=['w1', 'sigma'], compact=False, 
              coords={'w1_dim_0': 0, 'w1_dim_1': 0})
plt.suptitle('Trace Plots (first weight and sigma)', y=1.02)

# Plot 2: Posterior samples
ax2 = fig.add_subplot(gs[1, 0])
sample_indices = np.argsort(y_test)[:100]
n_plot_samples = min(100, predictions_original.shape[0])
for i in range(n_plot_samples):
    ax2.scatter(y_test[sample_indices], predictions_original[i][sample_indices],
               alpha=0.05, s=5, c='blue')
ax2.scatter(y_test[sample_indices], mean_pred[sample_indices],
           c='red', s=30, marker='x', linewidth=2, label='Posterior Mean', zorder=10)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'k--', linewidth=2, alpha=0.5)
ax2.set_xlabel('True Price ($100k)')
ax2.set_ylabel('Predicted Price ($100k)')
ax2.set_title('HMC Posterior Samples')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Predictions with credible intervals
ax3 = fig.add_subplot(gs[1, 1])
indices = np.random.choice(len(y_test), size=50, replace=False)
indices = indices[np.argsort(y_test[indices])]
ax3.fill_between(y_test[indices], lower_bound[indices], upper_bound[indices],
                 alpha=0.3, color='blue', label='95% Credible Interval')
ax3.scatter(y_test[indices], mean_pred[indices], c='red', s=30, zorder=10, 
           label='Posterior Mean')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'k--', linewidth=2, alpha=0.5)
ax3.set_xlabel('True Price ($100k)')
ax3.set_ylabel('Predicted Price ($100k)')
ax3.set_title('Predictions with 95% Credible Intervals')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Error vs Uncertainty
ax4 = fig.add_subplot(gs[1, 2])
errors = np.abs(mean_pred - y_test)
ax4.scatter(std_pred, errors, alpha=0.5, s=20)
ax4.set_xlabel('Epistemic Uncertainty')
ax4.set_ylabel('Absolute Error')
ax4.set_title('Error vs Uncertainty')
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
ax6.set_xlabel('Epistemic Uncertainty')
ax6.set_ylabel('Count')
ax6.set_title('Uncertainty Distribution')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Calibration
ax7 = fig.add_subplot(gs[2, 2])
confidence_levels = np.arange(0.1, 1.0, 0.1)
empirical_coverage = []
for conf in confidence_levels:
    alpha = (1 - conf) / 2
    lower = np.percentile(predictions_original, alpha * 100, axis=0)
    upper = np.percentile(predictions_original, (1 - alpha) * 100, axis=0)
    cov = np.mean((y_test >= lower) & (y_test <= upper))
    empirical_coverage.append(cov)

ax7.plot(confidence_levels, empirical_coverage, 'bo-', linewidth=2, 
        markersize=8, label='Empirical')
ax7.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect')
ax7.set_xlabel('Expected Coverage')
ax7.set_ylabel('Empirical Coverage')
ax7.set_title('Calibration Plot')
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.suptitle(f'PyMC HMC ({num_samples}×{num_chains} samples, {num_tune} tuning)',
            fontsize=16, fontweight='bold', y=0.998)
plt.tight_layout()
plt.show()

# Additional diagnostic plots using ArviZ
print("\nGenerating additional diagnostic plots...")

# Posterior plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot posterior distributions for some parameters
az.plot_posterior(trace, var_names=['sigma'], ax=axes[0, 0])
axes[0, 0].set_title('Posterior: Observation Noise (σ)')

# Energy plot (should look like half-normal)
az.plot_energy(trace, ax=axes[0, 1])

# Autocorrelation
az.plot_autocorr(trace, var_names=['sigma'], ax=axes[1, 0])

# Rank plot (should look uniform)
az.plot_rank(trace, var_names=['sigma'], ax=axes[1, 1])

plt.tight_layout()
plt.show()

# High uncertainty predictions
print(f'\n{"="*80}')
print("High Uncertainty Predictions (top 10):")
print("="*80)
print(f"{'Index':>5} | {'True':>8} | {'Predicted':>10} | {'Error':>8} | "
      f"{'Uncertainty':>11} | {'95% CI':>20}")
print("-"*80)
high_uncertainty_idx = np.argsort(std_pred)[-10:][::-1]
for idx in high_uncertainty_idx:
    error = mean_pred[idx] - y_test[idx]
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
print(f"Ratio (should be > 1): {high_unc_mae / low_unc_mae:.2f}")

print("\n" + "="*80)
print("PyMC HMC Analysis Complete!")
print("="*80)