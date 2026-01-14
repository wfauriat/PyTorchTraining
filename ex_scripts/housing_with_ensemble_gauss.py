import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

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

# Model that predicts BOTH mean and variance
class HeteroscedasticRegressor(nn.Module):
    """
    Neural network that predicts both mean and variance (aleatoric uncertainty)
    """
    def __init__(self):
        super().__init__()
        # Shared layers
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Separate heads for mean and log variance
        self.mean_head = nn.Linear(32, 1)
        self.log_var_head = nn.Linear(32, 1)
        
    def forward(self, x):
        # Shared representation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Mean prediction
        mean = self.mean_head(x).squeeze()
        
        # Log variance prediction (clamped for stability)
        log_var = self.log_var_head(x).squeeze()
        log_var = torch.clamp(log_var, min=-10, max=10)  # Prevent extreme values
        
        return mean, log_var

# Numerically stable Gaussian NLL loss
def gaussian_nll_loss(mean, log_var, target):
    """
    Numerically stable negative log-likelihood for Gaussian distribution
    Uses log_var directly to avoid exp() issues
    """
    # NLL = 0.5 * log(2π) + 0.5 * log_var + 0.5 * (target - mean)^2 / exp(log_var)
    # Rewrite: = 0.5 * log(2π) + 0.5 * log_var + 0.5 * (target - mean)^2 * exp(-log_var)
    
    squared_error = (target - mean) ** 2
    loss = 0.5 * log_var + 0.5 * squared_error * torch.exp(-log_var)
    return torch.mean(loss)

# Train ensemble with heteroscedastic models
def train_heteroscedastic_ensemble(n_models=20, epochs=200, lr=0.001):
    """
    Train ensemble where each model predicts mean and variance
    """
    models = []
    train_histories = []
    
    print(f"Training heteroscedastic ensemble of {n_models} models...")
    print("=" * 60)
    
    for model_idx in range(n_models):
        print(f"\nTraining Model {model_idx + 1}/{n_models}")
        print("-" * 60)
        
        # Create new model
        model = HeteroscedasticRegressor()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Track training history
        losses = []
        
        # Train
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            mean, log_var = model(X_train)
            
            # Check for NaN
            if torch.isnan(mean).any() or torch.isnan(log_var).any():
                print(f"  WARNING: NaN detected at epoch {epoch+1}, skipping this model")
                break
            
            loss = gaussian_nll_loss(mean, log_var, y_train_tensor)
            
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at epoch {epoch+1}, skipping this model")
                break
                
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 50 == 0:
                print(f'  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, '
                      f'Mean log_var: {log_var.mean().item():.4f}')
        
        # Only add model if training completed successfully
        if len(losses) == epochs:
            models.append(model)
            train_histories.append(losses)
            print(f"  Final Loss: {losses[-1]:.4f}")
        else:
            print(f"  Model training failed, not added to ensemble")
    
    print("\n" + "=" * 60)
    print(f"Ensemble training complete! Successfully trained {len(models)} models.")
    
    return models, train_histories

# Predict with heteroscedastic ensemble
def predict_with_heteroscedastic_ensemble(models, X, scaler_y):
    """
    Get predictions with BOTH epistemic and aleatoric uncertainty
    """
    if len(models) == 0:
        raise ValueError("No models in ensemble!")
    
    mean_predictions_scaled = []
    var_predictions_scaled = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            mean, log_var = model(X)
            
            # Clamp log_var for safety
            log_var = torch.clamp(log_var, min=-10, max=10)
            var = torch.exp(log_var)
            
            mean_predictions_scaled.append(mean.numpy())
            var_predictions_scaled.append(var.numpy())
    
    mean_predictions_scaled = np.array(mean_predictions_scaled)
    var_predictions_scaled = np.array(var_predictions_scaled)
    
    # Check for NaN
    if np.isnan(mean_predictions_scaled).any() or np.isnan(var_predictions_scaled).any():
        print("WARNING: NaN detected in predictions!")
    
    # Transform means to original scale
    individual_means = []
    for mean_scaled in mean_predictions_scaled:
        mean_original = scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        individual_means.append(mean_original)
    individual_means = np.array(individual_means)
    
    # Transform variances to original scale
    scale = scaler_y.scale_[0]
    var_predictions_original = var_predictions_scaled * (scale ** 2)
    
    # Ensemble mean
    ensemble_mean = individual_means.mean(axis=0)
    
    # Epistemic uncertainty (disagreement between models)
    epistemic_var = individual_means.var(axis=0)
    
    # Aleatoric uncertainty (average predicted variance)
    aleatoric_var = var_predictions_original.mean(axis=0)
    
    # Total uncertainty
    total_var = epistemic_var + aleatoric_var
    total_std = np.sqrt(total_var)
    
    return {
        'mean': ensemble_mean,
        'epistemic_std': np.sqrt(epistemic_var),
        'aleatoric_std': np.sqrt(aleatoric_var),
        'total_std': total_std,
        'individual_means': individual_means,
        'individual_vars': var_predictions_original
    }

# Train the ensemble with lower learning rate and gradient clipping
print("Starting training with numerical stability fixes...")
n_ensemble_models = 20
models, train_histories = train_heteroscedastic_ensemble(
    n_models=n_ensemble_models, epochs=200, lr=0.001  # Lower LR
)

if len(models) == 0:
    print("\nERROR: No models trained successfully!")
    print("This shouldn't happen with the fixes. Please check your data.")
    exit()

# Get predictions
print("\nGenerating predictions...")
results = predict_with_heteroscedastic_ensemble(models, X_test, scaler_y)

ensemble_mean = results['mean']
epistemic_std = results['epistemic_std']
aleatoric_std = results['aleatoric_std']
total_std = results['total_std']

# Check for NaN in results
if np.isnan(ensemble_mean).any():
    print("ERROR: NaN in predictions! Debugging info:")
    print(f"  NaN in epistemic_std: {np.isnan(epistemic_std).any()}")
    print(f"  NaN in aleatoric_std: {np.isnan(aleatoric_std).any()}")
    exit()

# Calculate metrics
mse = np.mean((ensemble_mean - y_test) ** 2)
mae = np.mean(np.abs(ensemble_mean - y_test))
r2 = 1 - np.sum((y_test - ensemble_mean) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

# Calculate coverage using total uncertainty
lower_bound = ensemble_mean - 1.96 * total_std
upper_bound = ensemble_mean + 1.96 * total_std
coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))

print(f'\n{"="*60}')
print("Heteroscedastic Ensemble Performance:")
print(f'{"="*60}')
print(f'Test MSE: {mse:.4f}')
print(f'Test MAE: {mae:.4f}')
print(f'Test R²:  {r2:.4f}')
print(f'95% CI Coverage: {coverage*100:.2f}%')
print(f'\nUncertainty Decomposition:')
print(f'  Mean Epistemic Std:  {epistemic_std.mean():.4f} (model uncertainty)')
print(f'  Mean Aleatoric Std:  {aleatoric_std.mean():.4f} (data noise)')
print(f'  Mean Total Std:      {total_std.mean():.4f}')
print(f'  Epistemic/Total:     {(epistemic_std.mean() / total_std.mean())*100:.1f}%')
print(f'  Aleatoric/Total:     {(aleatoric_std.mean() / total_std.mean())*100:.1f}%')


#%%


# Visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Plot 1: Training curves
ax1 = fig.add_subplot(gs[0, :])
for i, losses in enumerate(train_histories):
    ax1.plot(losses, alpha=0.7, linewidth=1.5, label=f'Model {i+1}')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Negative Log-Likelihood Loss')
ax1.set_title('Training Loss for Each Model')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Predictions with total uncertainty
ax2 = fig.add_subplot(gs[1, 0])
indices = np.random.choice(len(y_test), size=50, replace=False)
indices = indices[np.argsort(y_test[indices])]
ax2.errorbar(y_test[indices], ensemble_mean[indices], 
            yerr=1.96*total_std[indices],
            fmt='o', alpha=0.6, capsize=4, markersize=5, 
            label='Total Uncertainty', color='blue')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', linewidth=2, label='Perfect prediction')
ax2.set_xlabel('True Price ($100k)')
ax2.set_ylabel('Predicted Price ($100k)')
ax2.set_title('Predictions with Total Uncertainty (95% CI)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Epistemic vs Aleatoric
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(epistemic_std[indices], aleatoric_std[indices], alpha=0.5, s=20)
ax3.set_xlabel('Epistemic Uncertainty (Model)')
ax3.set_ylabel('Aleatoric Uncertainty (Data Noise)')
ax3.set_title('Epistemic vs Aleatoric Uncertainty')
# max_val = max(epistemic_std.max(), aleatoric_std.max())
# ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Uncertainty decomposition
ax4 = fig.add_subplot(gs[1, 2])
sample_indices = np.random.choice(len(y_test), size=50, replace=False)
x_pos = np.arange(len(sample_indices))
ax4.bar(x_pos, epistemic_std[sample_indices], alpha=0.7, 
       label='Epistemic', color='blue')
ax4.bar(x_pos, aleatoric_std[sample_indices], 
       bottom=epistemic_std[sample_indices],
       alpha=0.7, label='Aleatoric', color='orange')
ax4.set_xlabel('Sample Index')
ax4.set_ylabel('Standard Deviation')
ax4.set_title('Uncertainty Decomposition (50 samples)')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Total uncertainty vs error
ax5 = fig.add_subplot(gs[2, 0])
errors = np.abs(ensemble_mean - y_test)
ax5.scatter(total_std, errors, alpha=0.5, s=20, label='Total')
ax5.set_xlabel('Total Uncertainty (Std)')
ax5.set_ylabel('Absolute Error')
ax5.set_title('Error vs Uncertainty')
corr_total = np.corrcoef(total_std, errors)[0, 1]
ax5.text(0.05, 0.95, f'Correlation: {corr_total:.3f}',
        transform=ax5.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Predictions colored by epistemic
ax6 = fig.add_subplot(gs[2, 1])
scatter = ax6.scatter(y_test, ensemble_mean, c=epistemic_std,
                     cmap='Reds', alpha=0.6, s=20, vmin=0)
ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'k--', linewidth=2)
ax6.set_xlabel('True Price ($100k)')
ax6.set_ylabel('Predicted Price ($100k)')
ax6.set_title('Predictions (color = epistemic)')
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('Epistemic Std')
ax6.grid(True, alpha=0.3)

# Plot 7: Predictions colored by aleatoric
ax7 = fig.add_subplot(gs[2, 2])
scatter = ax7.scatter(y_test, ensemble_mean, c=aleatoric_std,
                     cmap='Blues', alpha=0.6, s=20, vmin=0)
ax7.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'k--', linewidth=2)
ax7.set_xlabel('True Price ($100k)')
ax7.set_ylabel('Predicted Price ($100k)')
ax7.set_title('Predictions (color = aleatoric)')
cbar = plt.colorbar(scatter, ax=ax7)
cbar.set_label('Aleatoric Std')
ax7.grid(True, alpha=0.3)

plt.suptitle(f'Heteroscedastic Deep Ensemble ({len(models)} models) - Uncertainty Decomposition',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("Analysis complete!")