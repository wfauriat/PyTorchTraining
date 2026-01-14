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

# Model definition
class HousePriceRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

# Train ensemble of models
def train_ensemble(n_models=20, epochs=100, lr=0.01):
    """
    Train an ensemble of models with different random initializations
    """
    models = []
    train_histories = []
    
    print(f"Training ensemble of {n_models} models...")
    print("=" * 60)
    
    for model_idx in range(n_models):
        print(f"\nTraining Model {model_idx + 1}/{n_models}")
        print("-" * 60)
        
        n_samples = len(X_train)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_train_boot = X_train[bootstrap_indices]
        y_train_boot = y_train_tensor[bootstrap_indices]

        # Create new model with random initialization
        model = HousePriceRegressor()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Track training history
        losses = []
        
        # Train
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_boot)
            loss = criterion(outputs, y_train_boot)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 50 == 0:
                print(f'  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        models.append(model)
        train_histories.append(losses)
        print(f"  Final Loss: {losses[-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("Ensemble training complete!")
    
    return models, train_histories

def predict_with_ensemble(models, X, scaler_y):
    """
    Get predictions and uncertainty from ensemble
    Correctly handles scaling by transforming before computing statistics
    """
    predictions_scaled = []
    
    # Collect predictions in scaled space
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X).numpy()
            predictions_scaled.append(pred)
    
    predictions_scaled = np.array(predictions_scaled)  # Shape: [n_models, n_samples]
    
    # Transform EACH model's predictions to original scale
    individual_preds = []
    for pred_scaled in predictions_scaled:
        pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        individual_preds.append(pred_original)
    
    individual_preds = np.array(individual_preds)  # Shape: [n_models, n_samples]
    
    # Compute statistics in ORIGINAL scale
    mean_pred = individual_preds.mean(axis=0)
    std_pred = individual_preds.std(axis=0)
    
    return mean_pred, std_pred, list(individual_preds)


# Train the ensemble
n_ensemble_models = 20
models, train_histories = train_ensemble(n_models=n_ensemble_models, epochs=200, lr=0.01)

#%%

# Get predictions with uncertainty
ensemble_mean, ensemble_std, individual_predictions = predict_with_ensemble(
    models, X_test, scaler_y
)

# Calculate metrics
mse = np.mean((ensemble_mean - y_test) ** 2)
mae = np.mean(np.abs(ensemble_mean - y_test))
r2 = 1 - np.sum((y_test - ensemble_mean) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

print(f'\n{"="*60}')
print("Ensemble Performance:")
print(f'{"="*60}')
print(f'Test MSE: {mse:.4f}')
print(f'Test MAE: {mae:.4f}')
print(f'Test R²:  {r2:.4f}')

# Visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training curves for all models
ax1 = fig.add_subplot(gs[0, :])
for i, losses in enumerate(train_histories):
    ax1.plot(losses, alpha=0.7, linewidth=1.5, label=f'Model {i+1}')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss for Each Model in Ensemble')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Individual model predictions vs ensemble mean
ax2 = fig.add_subplot(gs[1, 0])
# sample_indices = np.argsort(y_test)[:100]
sample_indices = np.random.randint(0, y_test.shape[0], 100)
for i, individual_pred in enumerate(individual_predictions):
    ax2.scatter(y_test[sample_indices], individual_pred[sample_indices], 
               alpha=0.3, s=10, label=f'Model {i+1}' if i < 3 else None)
ax2.scatter(y_test[sample_indices], ensemble_mean[sample_indices], 
           c='red', s=30, marker='x', linewidth=2, label='Ensemble Mean', zorder=10)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'k--', linewidth=2, alpha=0.5)
ax2.set_xlabel('True Price ($100k)')
ax2.set_ylabel('Predicted Price ($100k)')
ax2.set_title('Individual Models vs Ensemble (100 samples)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Ensemble predictions with error bars
ax3 = fig.add_subplot(gs[1, 1])

random_indices = np.random.choice(len(y_test), size=50, replace=False)
indices = random_indices[np.argsort(y_test[random_indices])]

# indices = np.argsort(y_test)[:50]
# indices = np.random.randint(0, y_test.shape[0], 100)
ax3.errorbar(y_test[indices], ensemble_mean[indices], 
            yerr=2*ensemble_std[indices],  # 2 std = ~95% confidence
            fmt='o', alpha=0.6, capsize=4, markersize=5, elinewidth=1.5)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', linewidth=2, label='Perfect prediction')
ax3.set_xlabel('True Price ($100k)')
ax3.set_ylabel('Predicted Price ($100k)')
ax3.set_title('Ensemble Predictions with 95% CI (50 samples)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Prediction error vs uncertainty
ax4 = fig.add_subplot(gs[1, 2])
errors = np.abs(ensemble_mean - y_test)
ax4.scatter(ensemble_std, errors, alpha=0.5, s=20)
ax4.set_xlabel('Uncertainty (Ensemble Std)')
ax4.set_ylabel('Absolute Error')
ax4.set_title('Prediction Error vs Uncertainty')
correlation = np.corrcoef(ensemble_std, errors)[0, 1]
ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
        transform=ax4.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax4.grid(True, alpha=0.3)

# Plot 5: Predictions colored by uncertainty
ax5 = fig.add_subplot(gs[2, 0])
scatter = ax5.scatter(y_test, ensemble_mean, c=ensemble_std, 
                     cmap='plasma', alpha=0.6, s=20)
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
ax6.hist(ensemble_std, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax6.axvline(ensemble_std.mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {ensemble_std.mean():.3f}')
ax6.set_xlabel('Uncertainty (Ensemble Std)')
ax6.set_ylabel('Count')
ax6.set_title('Uncertainty Distribution')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Agreement among models (variance decomposition)
ax7 = fig.add_subplot(gs[2, 2])
# For a few samples, show the distribution of predictions across models
sample_to_show = np.random.choice(len(y_test), 20, replace=False)
positions = []
values = []
for i, idx in enumerate(sample_to_show):
    preds = [ind_pred[idx] for ind_pred in individual_predictions]
    positions.extend([i] * len(preds))
    values.extend(preds)
    ax7.scatter([i], [y_test[idx]], color='red', marker='x', s=100, zorder=10)

ax7.scatter(positions, values, alpha=0.5, s=30)
ax7.set_xlabel('Sample Index')
ax7.set_ylabel('Predicted Price ($100k)')
ax7.set_title('Model Agreement (red X = true value)')
ax7.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Ensemble Uncertainty Quantification ({n_ensemble_models} models)', 
            fontsize=16, fontweight='bold', y=0.995)
plt.show()

# Print high uncertainty predictions
print(f'\n{"="*80}')
print("High Uncertainty Predictions (top 10):")
print("="*80)
print(f"{'Index':>5} | {'True':>8} | {'Predicted':>10} | {'Error':>8} | {'Uncertainty':>11} | {'Range':>15}")
print("-"*80)
high_uncertainty_idx = np.argsort(ensemble_std)[-10:][::-1]
for idx in high_uncertainty_idx:
    error = ensemble_mean[idx] - y_test[idx]
    pred_min = min([ind_pred[idx] for ind_pred in individual_predictions])
    pred_max = max([ind_pred[idx] for ind_pred in individual_predictions])
    print(f"{idx:>5} | ${y_test[idx]:>7.2f} | ${ensemble_mean[idx]:>9.2f} | "
          f"${error:>7.2f} | {ensemble_std[idx]:>11.3f} | "
          f"[${pred_min:.2f}, ${pred_max:.2f}]")

# Calibration check
median_uncertainty = np.median(ensemble_std)
low_unc_mask = ensemble_std < median_uncertainty
high_unc_mask = ensemble_std >= median_uncertainty

low_unc_mae = np.mean(np.abs(ensemble_mean[low_unc_mask] - y_test[low_unc_mask]))
high_unc_mae = np.mean(np.abs(ensemble_mean[high_unc_mask] - y_test[high_unc_mask]))

print(f'\n{"="*80}')
print("Calibration Check:")
print("="*80)
print(f"Low uncertainty samples MAE:  ${low_unc_mae:.4f}")
print(f"High uncertainty samples MAE: ${high_unc_mae:.4f}")
print(f"Ratio (should be > 1 if well-calibrated): {high_unc_mae / low_unc_mae:.2f}")

# Compare individual model performance
print(f'\n{"="*80}')
print("Individual Model Performance:")
print("="*80)
for i, individual_pred in enumerate(individual_predictions):
    model_mse = np.mean((individual_pred - y_test) ** 2)
    model_mae = np.mean(np.abs(individual_pred - y_test))
    print(f"Model {i+1}: MSE={model_mse:.4f}, MAE={model_mae:.4f}")
print(f"\nEnsemble:  MSE={mse:.4f}, MAE={mae:.4f}")
print(f"Improvement over average single model: {((np.mean([np.mean((ind - y_test)**2) for ind in individual_predictions]) - mse) / mse * 100):.1f}%")
# %%
