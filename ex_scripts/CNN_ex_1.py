#%% ################################
## IMPORTS
####################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# import torchvision
import torchvision.transforms as T

from torchvision.datasets import CIFAR10

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional
# import math


#%% ################################
## DEVICE SET UP
####################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")


#%% ################################
##LOADING DATASETS
####################################

# CIFAR-10 transforms
train_transform = T.Compose([
    T.RandomCrop(32, padding=4), # Creates slight translations/shifts of the image, helping the model learn position-invariant features
    T.RandomHorizontalFlip(), # Data augmentation that doubles your effective dataset size and helps with left-right symmetry
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Load datasets
train_dataset = CIFAR10('../data', train=True, download=True, 
                        transform=train_transform)
test_dataset = CIFAR10('../data', train=False, download=True, 
                       transform=test_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=128, 
                          shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, 
                         shuffle=False, num_workers=2, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')


#%% ################################
## VIEW IMAGES FROM DATASET (RGB with Pyplot)
####################################


def show_samples(dataset, num_samples=3):
    _, axes = plt.subplots(num_samples, num_samples, figsize=(8, 8))

    rndidx = torch.randint(0, len(dataset), (num_samples,)*2).flatten()
    
    # Denormalize for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    
    for i, ax in enumerate(axes.flatten()):
        img, label = dataset[rndidx[i]]
        img = img * std + mean  # Denormalize
        img = img.permute(1, 2, 0).clamp(0, 1) # Set in (32 x 32 x 3 - RGB)
        ax.imshow(img)
        ax.set_title(classes[label])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

show_samples(test_dataset)


#%% ################################
## DEFINITION OF NETWORK ARCHITECTURE
####################################

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture similar to LeNet.
    Architecture: Conv -> Pool -> Conv -> Pool -> FC -> FC -> Output
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pools of 2x2 on 32x32 input: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten
        x = x.flatten(1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Test
model = SimpleCNN(num_classes=10)
x = torch.randn(1, 3, 32, 32)
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


#%% ################################
## TRAINING LOOP DEFINITION
####################################


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


#%% ################################
## TRAINING 
####################################

model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                             lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Training history
history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}


### START OF TRAINING
num_epochs = 20
print(f"Training CNN for {num_epochs} epochs...")

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader,
                                         criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    print(f"Epoch {epoch+1:2d}/{num_epochs} | "
          f"Train Loss: {train_loss:.3f} Acc: {train_acc:.1f}% | "
          f"Test Loss: {test_loss:.3f} Acc: {test_acc:.1f}% | "
          f"LR: {scheduler.get_last_lr()[0]:.4f}")


#%% ################################
## VISUALISATION OF TRAINING RESULTS
####################################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['test_loss'], label='Test')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves')
ax1.legend()
ax1.grid(True)

ax2.plot(history['train_acc'], label='Train')
ax2.plot(history['test_acc'], label='Test')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy Curves')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


#%% ################################
## TESTING ONE STEP OF TRAINING
####################################

train_loss, train_acc = train_epoch(model, train_loader,
                                         criterion, optimizer, device)
print(f"Train Loss: {train_loss:.3f} Acc: {train_acc:.1f}%")
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.3f} Acc: {test_acc:.1f}%")


#%% ################################
## TEST PREDICTION ON A BATCH
####################################

test_loader2 = DataLoader(test_dataset, batch_size=128, 
                         shuffle=True, num_workers=2, pin_memory=True)

data_iter = iter(test_loader2)
img, label = next(data_iter)

model.eval() 
with torch.no_grad():
    output = model(img.to(device))

test_labels = [classes[i] for i in output.argmax(1)]
true_labels = [classes[i] for i in label]


for pred, true in zip(test_labels, true_labels):
    print(f"Predicted: {pred:10s} | True: {true}")

correct = sum(pred == true for pred, true in zip(test_labels, true_labels))
total = len(test_labels)
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

# Display first 10 images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i in range(10):
    img[i] = img[i] * std + mean 
    axes[i].imshow(img[i].permute(1, 2, 0).clamp(0,1))  # Change from CxHxW to HxWxC
    axes[i].set_title(f"True: {classes[label[i]]} | Pred: {test_labels[i]} ")
    axes[i].axis('off')
plt.tight_layout()
plt.show()