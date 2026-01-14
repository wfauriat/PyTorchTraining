import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load data (same as before)
housing = fetch_california_housing()
X = housing.data
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=43)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
X_train = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test_scaled)

# Model with dropout
class HousePriceRegressorUncertain(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()

# Train
model = HousePriceRegressorUncertain(dropout_rate=0.3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 40 == 0:
        print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')

# MC Dropout for uncertainty
def predict_with_uncertainty_regression(model, X, scaler_y, n_samples=100):
    """
    Run multiple forward passes to estimate prediction uncertainty
    """
    model.train()  # Keep dropout active
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(X)
            predictions.append(output.numpy())
    
    predictions = np.array(predictions)  # Shape: [n_samples, n_test]
    
    # Statistics in scaled space
    mean_scaled = predictions.mean(axis=0)
    std_scaled = predictions.std(axis=0)
    
    # Transform back to original scale
    mean_pred = scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
    # Approximate std in original scale (simplified)
    std_pred = std_scaled * scaler_y.scale_[0]
    
    return mean_pred, std_pred

# Get predictions with uncertainty
predictions, uncertainties = predict_with_uncertainty_regression(
    model, X_test, scaler_y, n_samples=100
)

# Calculate metrics
mse = np.mean((predictions - y_test) ** 2)
mae = np.mean(np.abs(predictions - y_test))
print(f'\nTest MSE: {mse:.4f}')
print(f'Test MAE: {mae:.4f}')

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Predictions vs True with Error Bars
indices = np.argsort(y_test)[:50]  # Show first 50 sorted samples
axes[0, 0].errorbar(y_test[indices], predictions[indices], 
                    yerr=2*uncertainties[indices],  # 2 std = ~95% confidence
                    fmt='o', alpha=0.6, capsize=3, markersize=4)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect prediction')
axes[0, 0].set_xlabel('True Price ($100k)')
axes[0, 0].set_ylabel('Predicted Price ($100k)')
axes[0, 0].set_title('Predictions with 95% Confidence Intervals (50 samples)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Prediction Error vs Uncertainty
errors = np.abs(predictions - y_test)
axes[0, 1].scatter(uncertainties, errors, alpha=0.5, s=20)
axes[0, 1].set_xlabel('Uncertainty (std)')
axes[0, 1].set_ylabel('Absolute Error')
axes[0, 1].set_title('Prediction Error vs Uncertainty')
axes[0, 1].grid(True, alpha=0.3)

# Add correlation info
correlation = np.corrcoef(uncertainties, errors)[0, 1]
axes[0, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=axes[0, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Predictions colored by uncertainty
scatter = axes[1, 0].scatter(y_test, predictions, c=uncertainties, 
                             cmap='viridis', alpha=0.6, s=20)
axes[1, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', linewidth=2)
axes[1, 0].set_xlabel('True Price ($100k)')
axes[1, 0].set_ylabel('Predicted Price ($100k)')
axes[1, 0].set_title('Predictions (color = uncertainty)')
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_label('Uncertainty')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Uncertainty distribution
axes[1, 1].hist(uncertainties, bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[1, 1].axvline(uncertainties.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {uncertainties.mean():.3f}')
axes[1, 1].set_xlabel('Uncertainty (std)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Uncertainty Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print high uncertainty predictions
print("\nHigh Uncertainty Predictions (top 10):")
print("-" * 70)
print(f"{'Index':>5} | {'True':>8} | {'Predicted':>10} | {'Error':>8} | {'Uncertainty':>11}")
print("-" * 70)
high_uncertainty_idx = np.argsort(uncertainties)[-10:][::-1]
for idx in high_uncertainty_idx:
    error = predictions[idx] - y_test[idx]
    print(f"{idx:>5} | ${y_test[idx]:>7.2f} | ${predictions[idx]:>9.2f} | ${error:>7.2f} | {uncertainties[idx]:>11.3f}")

# Calibration check: are high uncertainty predictions actually worse?
# Split into low/high uncertainty groups
median_uncertainty = np.median(uncertainties)
low_unc_mask = uncertainties < median_uncertainty
high_unc_mask = uncertainties >= median_uncertainty

low_unc_mae = np.mean(np.abs(predictions[low_unc_mask] - y_test[low_unc_mask]))
high_unc_mae = np.mean(np.abs(predictions[high_unc_mask] - y_test[high_unc_mask]))

print(f"\nCalibration Check:")
print(f"Low uncertainty samples MAE:  ${low_unc_mae:.4f}")
print(f"High uncertainty samples MAE: ${high_unc_mae:.4f}")
print(f"Ratio (should be > 1 if well-calibrated): {high_unc_mae / low_unc_mae:.2f}")