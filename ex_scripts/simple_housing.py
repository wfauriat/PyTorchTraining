#%%

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np

#%%

# Load data
housing = fetch_california_housing()
X = housing.data  # 8 features: median income, house age, etc.
y = housing.target  # Target: median house value

# Split and normalize
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                         test_size=0.2, random_state=42)
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


#%%

# Regression model
class HousePriceRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Single output for regression
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()  # Remove extra dimension

#%%

# Training
model = HousePriceRegressor()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Track metrics
train_losses = []
train_r2_scores = []
epochs_list = []

model.train()
for epoch in range(200):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train_tensor)
    
    # Calculate R² score (coefficient of determination)
    with torch.no_grad():
        ss_res = torch.sum((y_train_tensor - outputs) ** 2)
        ss_tot = torch.sum((y_train_tensor - y_train_tensor.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Store metrics
    train_losses.append(loss.item())
    train_r2_scores.append(r2.item())
    epochs_list.append(epoch + 1)
    
    if (epoch + 1) % 40 == 0:
        print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}, R²: {r2.item():.4f}')

# Evaluate on test set
model.eval()
with torch.no_grad():
    predictions_scaled = model(X_test)
    test_loss = criterion(predictions_scaled, y_test_tensor)
    
    # Calculate test R²
    ss_res = torch.sum((y_test_tensor - predictions_scaled) ** 2)
    ss_tot = torch.sum((y_test_tensor - y_test_tensor.mean()) ** 2)
    test_r2 = 1 - ss_res / ss_tot
    
    # Inverse transform to original scale
    predictions = scaler_y.inverse_transform(predictions_scaled.numpy().reshape(-1, 1)).flatten()
    
    print(f'\nTest MSE: {test_loss.item():.4f}')
    print(f'Test R²: {test_r2.item():.4f}')

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training Loss
axes[0, 0].plot(epochs_list, train_losses, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].set_title('Training Loss over Time')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training R² Score
axes[0, 1].plot(epochs_list, train_r2_scores, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].set_title('Training R² Score over Time')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect fit')
axes[0, 1].legend()

# Plot 3: Predictions vs True Values (Scatter)
axes[1, 0].scatter(y_test, predictions, alpha=0.5, s=20)
axes[1, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect prediction')
axes[1, 0].set_xlabel('True Price ($100k)')
axes[1, 0].set_ylabel('Predicted Price ($100k)')
axes[1, 0].set_title('Predictions vs True Values (Test Set)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Residuals (Prediction Error)
residuals = predictions - y_test
axes[1, 1].scatter(predictions, residuals, alpha=0.5, s=20)
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Predicted Price ($100k)')
axes[1, 1].set_ylabel('Residual (Predicted - True)')
axes[1, 1].set_title('Residual Plot')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print sample predictions
print("\nSample Predictions (first 10 test samples):")
print("-" * 60)
print(f"{'True Price':>12} | {'Predicted':>12} | {'Error':>12}")
print("-" * 60)
for i in range(min(10, len(y_test))):
    error = predictions[i] - y_test[i]
    print(f"${y_test[i]:>10.2f} | ${predictions[i]:>10.2f} | ${error:>10.2f}")