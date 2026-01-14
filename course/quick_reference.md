# PyTorch Quick Reference

> One-page cheat sheet for PyTorch fundamentals

---

## Tensors

### Creation
```python
torch.tensor([1, 2, 3])              # From list
torch.zeros(3, 4)                    # All zeros
torch.ones(3, 4)                     # All ones
torch.rand(3, 4)                     # Uniform [0, 1)
torch.randn(3, 4)                    # Normal N(0,1)
torch.arange(0, 10, 2)              # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)             # [0.0, 0.25, 0.5, 0.75, 1.0]
torch.eye(3)                         # Identity matrix
```

### Shapes
```python
x.shape, x.size()                    # Get shape
x.view(2, -1)                        # Reshape (contiguous required)
x.reshape(2, -1)                     # Reshape (copies if needed)
x.squeeze()                          # Remove size-1 dims
x.unsqueeze(0)                       # Add dim at position 0
x.transpose(0, 1)                    # Swap dims
x.permute(2, 0, 1)                  # Reorder dims
x.flatten()                          # Flatten to 1D
```

### Operations
```python
x + y, x - y, x * y, x / y          # Element-wise
x @ y, torch.matmul(x, y)           # Matrix multiply
x.sum(), x.mean(), x.std()          # Reductions
x.max(), x.argmax()                 # Max value and index
torch.cat([x, y], dim=0)            # Concatenate
torch.stack([x, y], dim=0)          # Stack (new dim)
```

### Device
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)                     # Move to device
x = x.cuda()                         # Move to GPU
x = x.cpu()                          # Move to CPU
```

---

## Autograd

### Basics
```python
x = torch.tensor(2.0, requires_grad=True)   # Enable gradients
y = x ** 2                                   # Build computation graph
y.backward()                                 # Compute gradients
print(x.grad)                                # Access gradient
x.grad.zero_()                               # Zero gradients
```

### Context Managers
```python
with torch.no_grad():                # Disable gradient tracking
    y = model(x)

with torch.inference_mode():         # Faster inference mode
    y = model(x)

with torch.enable_grad():            # Re-enable gradients
    y = model(x)
```

### Utilities
```python
x.detach()                           # Detach from graph
torch.autograd.grad(y, x)           # Manual gradient computation
torch.autograd.detect_anomaly()     # Debug NaN/Inf
```

---

## Modules

### Basic Module
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)

    def forward(self, x):
        return self.layer(x)

model = MyModel()
output = model(input)  # Always call model(x), not model.forward(x)
```

### Common Layers
```python
nn.Linear(in_features, out_features)          # Fully connected
nn.Conv2d(in_ch, out_ch, kernel_size)        # 2D convolution
nn.MaxPool2d(kernel_size)                     # Max pooling
nn.BatchNorm2d(num_features)                  # Batch normalization
nn.Dropout(p=0.5)                             # Dropout
nn.ReLU(), nn.Tanh(), nn.Sigmoid()           # Activations
nn.LSTM(input_size, hidden_size)             # LSTM
nn.Embedding(num_embeddings, embedding_dim)  # Embeddings
```

### Containers
```python
nn.Sequential(layer1, layer2, layer3)        # Sequential
nn.ModuleList([layer1, layer2])              # List of modules
nn.ModuleDict({'encoder': enc, 'decoder': dec})  # Dict of modules
```

### Parameters
```python
model.parameters()                    # All parameters
model.named_parameters()              # With names
for param in model.parameters():
    param.requires_grad = False       # Freeze
```

### State
```python
model.train()                         # Training mode
model.eval()                          # Evaluation mode
torch.save(model.state_dict(), 'model.pth')   # Save
model.load_state_dict(torch.load('model.pth'))  # Load
```

---

## Training Loop

### Standard Loop
```python
for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()          # 1. Zero gradients
        output = model(data)           # 2. Forward pass
        loss = criterion(output, target)  # 3. Compute loss
        loss.backward()                # 4. Backward pass
        optimizer.step()               # 5. Update weights
```

### Loss Functions
```python
nn.CrossEntropyLoss()                 # Multi-class (logits → classes)
nn.BCEWithLogitsLoss()                # Binary/multi-label (logits → 0/1)
nn.MSELoss()                          # Regression (L2)
nn.L1Loss()                           # Regression (L1)
```

### Optimizers
```python
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optim.Adam(model.parameters(), lr=0.001)
optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Learning Rate Schedulers
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)

# After epoch
scheduler.step()

# For ReduceLROnPlateau
scheduler.step(val_loss)
```

### Gradient Clipping
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Data Loading

### Dataset
```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(data, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

---

## Common Patterns

### Device-Agnostic Code
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
for data, target in loader:
    data, target = data.to(device), target.to(device)
```

### Evaluation
```python
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        # Compute metrics
```

### Saving/Loading Checkpoint
```python
# Save
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Debugging

### Check Gradients
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### Detect Anomalies
```python
with torch.autograd.detect_anomaly():
    loss.backward()  # Will error at NaN source
```

### Hooks
```python
def hook_fn(module, input, output):
    print(f"Output shape: {output.shape}")

handle = model.layer.register_forward_hook(hook_fn)
model(x)
handle.remove()
```

---

## Common Mistakes

### ❌ Forgetting to zero gradients
```python
# Wrong
output = model(x)
loss.backward()
optimizer.step()

# Correct
optimizer.zero_grad()
output = model(x)
loss.backward()
optimizer.step()
```

### ❌ Using Python list for modules
```python
# Wrong - parameters not registered
self.layers = [nn.Linear(10, 10) for _ in range(5)]

# Correct
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
```

### ❌ Calling forward directly
```python
# Wrong
output = model.forward(x)

# Correct
output = model(x)
```

### ❌ Not setting eval mode
```python
# Wrong - dropout/batchnorm still in training mode
predictions = model(test_data)

# Correct
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

---

## Performance Tips

1. **Use DataLoader with multiple workers**: `num_workers=4`
2. **Pin memory for GPU**: `pin_memory=True` in DataLoader
3. **Use in-place operations when safe**: `x += 1` instead of `x = x + 1`
4. **Vectorize operations**: Avoid Python loops over tensors
5. **Profile code**: Use `torch.profiler` to find bottlenecks
6. **Mixed precision**: Use `torch.cuda.amp` on modern GPUs
7. **Gradient accumulation**: Simulate larger batches
8. **Compile model (PyTorch 2.0+)**: `model = torch.compile(model)`

---

## Useful Links

- [Official Documentation](https://pytorch.org/docs/)
- [Tutorials](https://pytorch.org/tutorials/)
- [Forums](https://discuss.pytorch.org/)

**Back to**: [README](README.md) | [01_tensors.md](01_tensors.md) | [02_autograd.md](02_autograd.md) | [03_modules.md](03_modules.md) | [04_training.md](04_training.md)
