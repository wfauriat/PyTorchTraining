#%% ################################
## IMPORTS
####################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from CNN_archi import SimpleCNN, VGGStyleCNN, ResNet, BasicBlock, Bottleneck

# import torchvision
import torchvision.transforms as T

from torchvision.datasets import CIFAR10, MNIST

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional
# import math

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


# sel_case = "MNIST"
sel_case = "CIFAR-10"


#%% ################################
## DEVICE SET UP
####################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")


#%% ################################
##LOADING MNIST DATASET
####################################

if sel_case=="MNIST":

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    train_dataset =MNIST(root='../data', train=True,
        download=True, transform=transform)
    test_dataset = MNIST(root='../data', train=False, 
        download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64,
        shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64,
        shuffle=False, num_workers=2, pin_memory=True
    )

classes = [str(el) for el in list(range(10))]


#%% ################################
##LOADING CIFAR DATASET
####################################

if sel_case=="CIFAR-10":

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
## VIEW IMAGES FROM CIFAR DATASET (RGB with Pyplot)
####################################

if sel_case=="CIFAR-10":

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
## VIEW IMAGES FROM MNIST DATASET
####################################

if sel_case=="MNIST":

    # Get a batch
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Display first 10 images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')  # squeeze removes channel dimension
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


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

if sel_case == "CIFAR-10":
    # model = SimpleCNN(dimw=32, num_classes=10, num_channels=3).to(device)
    model = VGGStyleCNN(num_classes=10).to(device)
if sel_case == "MNIST":
    model = SimpleCNN(dimw=28, num_classes=10, num_channels=1).to(device)



# model = ResNet18(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                             lr=0.1, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
    
print(" Number of model parameters : " + \
      f"{sum([m.numel() for m in model.parameters()]):}")


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

# train_loss, train_acc = train_epoch(model, train_loader,
#                                          criterion, optimizer, device)
# print(f"Train Loss: {train_loss:.3f} Acc: {train_acc:.1f}%")
# test_loss, test_acc = evaluate(model, test_loader, criterion, device)
# print(f"Test Loss: {test_loss:.3f} Acc: {test_acc:.1f}%")


#%% ################################
## TEST PREDICTION ON A BATCH
####################################

test_loader2 = DataLoader(test_dataset, batch_size=1024, 
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
_, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
for i in range(10):
    if sel_case == "CIFAR-10":
        img[i] = img[i] * std + mean
        axes[i].imshow(img[i].permute(1, 2, 0).clamp(0,1))  # Change from CxHxW to HxWxC
    if sel_case == "MNIST":
        axes[i].imshow(img[i].squeeze(), cmap="gray")
    axes[i].set_title(f"True: {classes[label[i]]} | Pred: {test_labels[i]} ")
    axes[i].axis('off')
plt.tight_layout()
plt.show()

#%% ################################
## VISUALIZE FAILED PREDICTIONS 
####################################

maskdif = [el1 != el2 for (el1, el2) in zip(test_labels,true_labels)]
diffs = np.arange(len(true_labels))[maskdif]

_, ax = plt.subplots(1, 5, figsize=(14,3))

for i, id in enumerate(diffs[1:5]):
    print(id)
    if sel_case == "CIFAR-10":
        img[id] = img[id] * std + mean
        ax[i].imshow(img[id].permute(1, 2, 0).clamp(0,1))  # Change from CxHxW to HxWxC
    if sel_case == "MNIST":
        ax[i].imshow(img[id].squeeze(), cmap="gray")
    ax[i].set_title(f"True: {classes[label[id]]} | Pred: {test_labels[id]} ")
    ax[i].axis('off')

plt.tight_layout()
plt.show()
# %%
