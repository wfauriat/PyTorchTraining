#%%

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np

#%%

# Load data
iris = load_iris()
X = iris.data  # 4 features: sepal length, sepal width, petal length, petal width
y = iris.target  # 3 classes: setosa, versicolor, virginica

# Split and normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                    random_state=np.random.randint(100))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

#%%

# Simple neural network
class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)   # 4 input features
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)     # 3 output classes
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
#%%

# Training
model = IrisClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Track metrics
train_losses = []
train_accuracies = []
epochs_list = []

model.train()
for epoch in range(200):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Calculate training accuracy
    _, predicted = torch.max(outputs, 1)
    train_acc = (predicted == y_train).sum().item() / len(y_train) * 100
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Store metrics
    train_losses.append(loss.item())
    train_accuracies.append(train_acc)
    epochs_list.append(epoch + 1)
    
    if (epoch + 1) % 40 == 0:
        print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}, Train Acc: {train_acc:.2f}%')

# Evaluate on test set
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    test_accuracy = (predicted == y_test).sum().item() / len(y_test) * 100
    print(f'\nTest Accuracy: {test_accuracy:.2f}%')


#%%


# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training Loss
axes[0].plot(epochs_list, train_losses, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss over Time')
axes[0].grid(True, alpha=0.3)

# Plot 2: Training Accuracy
axes[1].plot(epochs_list, train_accuracies, 'g-', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Training Accuracy over Time')
axes[1].grid(True, alpha=0.3)

# Plot 3: Predictions vs Truth (Confusion-style visualization)
class_names = iris.target_names
predicted_np = predicted.numpy()
y_test_np = y_test.numpy()

# Create a scatter plot showing predictions vs true labels
axes[2].scatter(y_test_np, predicted_np, alpha=0.6, s=100)
axes[2].plot([0, 2], [0, 2], 'r--', linewidth=2, label='Perfect prediction')
axes[2].set_xlabel('True Class')
axes[2].set_ylabel('Predicted Class')
axes[2].set_title('Predictions vs True Labels (Test Set)')
axes[2].set_xticks([0, 1, 2])
axes[2].set_yticks([0, 1, 2])
axes[2].set_xticklabels(class_names, rotation=45)
axes[2].set_yticklabels(class_names)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print detailed results
print("\nDetailed Test Results:")
print("-" * 50)
for i in range(len(y_test)):
    true_class = class_names[y_test[i]]
    pred_class = class_names[predicted[i]]
    status = "✓" if y_test[i] == predicted[i] else "✗"
    print(f"{status} Sample {i+1:2d}: True={true_class:12s} | Pred={pred_class:12s}")