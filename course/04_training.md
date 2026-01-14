# Training Loop Mechanics

> **Core Question**: How do we train neural networks in PyTorch?

**In this guide:**
- [Overview](#overview)
- [Anatomy of a Training Loop](#anatomy-of-a-training-loop)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Learning Rate Scheduling](#learning-rate-scheduling)
- [Gradient Clipping](#gradient-clipping)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)

---

## Overview

Training a neural network involves:
1. **Forward pass**: Compute predictions
2. **Loss computation**: Measure error
3. **Backward pass**: Compute gradients
4. **Parameter update**: Adjust weights

PyTorch gives you full control over this process, unlike high-level frameworks that hide the training loop.

---

## Anatomy of a Training Loop

### The Five Essential Steps

```python
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass
        output = model(data)

        # 3. Compute loss
        loss = criterion(output, target)

        # 4. Backward pass
        loss.backward()

        # 5. Update parameters
        optimizer.step()
```

### Why Each Step Matters

| Step | Purpose | What Happens If Skipped |
|------|---------|------------------------|
| **zero_grad()** | Clear old gradients | Gradients accumulate → wrong updates |
| **forward** | Get predictions | No output to compute loss |
| **loss** | Measure error | No signal for improvement |
| **backward()** | Compute gradients | No gradients → no learning |
| **step()** | Update weights | Weights don't change → no learning |

### Complete Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Five essential steps
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Log epoch statistics
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')
```

---

## Loss Functions

The loss function measures how wrong your predictions are.

### Classification Losses

#### Cross-Entropy Loss

**For multi-class classification** (mutually exclusive classes):

```python
criterion = nn.CrossEntropyLoss()

# Model outputs logits (raw scores)
logits = model(x)  # Shape: (batch_size, num_classes)
loss = criterion(logits, targets)  # targets: class indices

# Combines nn.LogSoftmax + nn.NLLLoss
```

**How it works**:
```python
# Equivalent to:
log_probs = F.log_softmax(logits, dim=1)
loss = F.nll_loss(log_probs, targets)
```

**When to use**: Standard multi-class classification (MNIST, ImageNet, etc.)

#### Binary Cross-Entropy

**For binary classification** or **multi-label** classification:

```python
criterion = nn.BCEWithLogitsLoss()  # Preferred (numerically stable)

# Model outputs logits
logits = model(x)  # Shape: (batch_size, 1) or (batch_size, num_labels)
loss = criterion(logits, targets)  # targets: 0 or 1

# Combines sigmoid + BCE
```

**Multi-label example**:
```python
# Each sample can have multiple labels
logits = model(x)  # (batch, 10) - 10 possible labels
targets = torch.tensor([[1, 0, 1, 0, 0, 1, 0, 0, 0, 1]])  # Multiple 1s
loss = criterion(logits, targets.float())
```

### Regression Losses

#### Mean Squared Error (MSE)

**For regression** (continuous values):

```python
criterion = nn.MSELoss()

predictions = model(x)
loss = criterion(predictions, targets)

# loss = mean((predictions - targets)^2)
```

#### Mean Absolute Error (MAE)

**More robust to outliers**:

```python
criterion = nn.L1Loss()

loss = criterion(predictions, targets)
# loss = mean(|predictions - targets|)
```

#### Smooth L1 Loss

**Hybrid: L2 for small errors, L1 for large** (used in object detection):

```python
criterion = nn.SmoothL1Loss()

# L2 when |error| < 1, otherwise L1
```

### Custom Loss Functions

```python
def custom_loss(predictions, targets):
    mse = F.mse_loss(predictions, targets)
    mae = F.l1_loss(predictions, targets)
    return mse + 0.1 * mae  # Weighted combination

# Use like any criterion
loss = custom_loss(model(x), y)
loss.backward()
```

### Loss Reduction

```python
# Default: mean over batch
criterion = nn.MSELoss(reduction='mean')

# Sum over batch
criterion = nn.MSELoss(reduction='sum')

# No reduction (per-sample loss)
criterion = nn.MSELoss(reduction='none')
losses = criterion(pred, target)  # Shape: (batch_size,)
# Can now weight samples differently
weighted_loss = (losses * sample_weights).mean()
```

---

## Optimizers

Optimizers update model parameters using gradients.

### SGD (Stochastic Gradient Descent)

**Basic optimizer**:

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)

# With momentum (helps acceleration)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# With weight decay (L2 regularization)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
```

**Update rule**:
```
θ_{t+1} = θ_t - lr * ∇θ
```

**With momentum**:
```
v_{t+1} = momentum * v_t + ∇θ
θ_{t+1} = θ_t - lr * v_{t+1}
```

### Adam (Adaptive Moment Estimation)

**Most popular optimizer** (adaptive learning rates):

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

**Pros**:
- Adapts learning rate per parameter
- Works well with default settings
- Handles sparse gradients well

**Cons**:
- Can overfit on small datasets
- May not converge as well as SGD with momentum on some tasks

### AdamW (Adam with Weight Decay)

**Adam with proper weight decay** (recommended over Adam):

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Why AdamW > Adam**: Decoupled weight decay (better generalization).

### Other Optimizers

```python
# RMSprop (good for RNNs)
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Adagrad (adaptive, but lr decreases over time)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# RAdam (rectified Adam, more stable)
optimizer = optim.RAdam(model.parameters(), lr=0.001)
```

### Optimizer Comparison

| Optimizer | When to Use | Typical LR |
|-----------|------------|------------|
| **SGD + momentum** | Vision models, when you can tune LR carefully | 0.1 - 0.01 |
| **Adam** | General purpose, quick prototyping | 0.001 - 0.0001 |
| **AdamW** | Transformers, NLP, when weight decay matters | 0.001 - 0.0001 |
| **RMSprop** | RNNs, online learning | 0.001 |

### Parameter Groups (Different Learning Rates)

```python
# Different LR for different parts of model
optimizer = optim.Adam([
    {'params': model.base.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

**Common in**:
- Transfer learning (freeze early layers, faster LR for new layers)
- Fine-tuning pretrained models

---

## Learning Rate Scheduling

Adjust learning rate during training for better convergence.

### Why Schedule LR?

- **High LR early**: Fast initial progress
- **Low LR later**: Fine-tuning, stability

### Common Schedulers

#### StepLR

**Decay LR by gamma every N epochs**:

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()  # Update LR after each epoch
```

**Example**: LR = 0.1 → 0.01 (epoch 30) → 0.001 (epoch 60)

#### CosineAnnealingLR

**Smoothly decay LR following cosine curve**:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

**Shape**:
```
LR
 │ \
 │  \
 │   \_____
 └────────── Epochs
```

#### OneCycleLR

**Cycle LR up then down** (popularized by fast.ai):

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs
)

for epoch in range(num_epochs):
    for batch in train_loader:
        train_step(...)
        scheduler.step()  # Update every batch!
```

**Shape**:
```
LR
 │   /\
 │  /  \
 │ /    \___
 └────────── Steps
```

#### ReduceLROnPlateau

**Reduce LR when metric stops improving**:

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)

for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # Pass metric to scheduler
```

**Adaptive**: Waits `patience` epochs, then reduces LR by `factor` if no improvement.

### Scheduler Best Practices

```python
# Pattern 1: Per-epoch scheduling
for epoch in range(num_epochs):
    train_one_epoch()
    scheduler.step()

# Pattern 2: Per-batch scheduling (OneCycleLR)
for epoch in range(num_epochs):
    for batch in train_loader:
        train_step()
        scheduler.step()

# Pattern 3: Metric-based (ReduceLROnPlateau)
for epoch in range(num_epochs):
    train_one_epoch()
    val_metric = validate()
    scheduler.step(val_metric)
```

---

## Gradient Clipping

Prevent exploding gradients by limiting gradient magnitude.

### Why Clip Gradients?

- **RNNs**: Prone to exploding gradients
- **Deep networks**: Can have unstable gradients
- **Large learning rates**: Prevent divergence

### Clip by Norm

**Most common approach**:

```python
# After loss.backward(), before optimizer.step()
loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

**What it does**: Scales gradients if their norm exceeds `max_norm`.

### Clip by Value

**Clamp each gradient element**:

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**Less common** (clip_grad_norm_ is usually better).

### When to Use

```python
# Good default for RNNs
clip_grad_norm_(model.parameters(), max_norm=1.0)

# Transformers (larger networks, more stable)
clip_grad_norm_(model.parameters(), max_norm=5.0)

# If unsure, monitor gradient norms first
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm}")
```

---

## Best Practices

### 1. Reproducibility

```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
```

### 2. Mixed Precision Training

**Faster training with less memory** (on modern GPUs):

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    # Forward in mixed precision
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Scaled backward
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Gradient Accumulation

**Simulate larger batch sizes**:

```python
accumulation_steps = 4

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps  # Scale loss

    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Why**: When GPU memory is limited, accumulate gradients over multiple batches.

### 4. Learning Rate Warm-up

**Gradually increase LR at start** (common for Transformers):

```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.base_lr * min(self.step_num / self.warmup_steps, 1.0)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### 5. Early Stopping

**Stop training when validation performance plateaus**:

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## Common Patterns

### Pattern 1: Train and Validation Loop

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(val_loader)

    return avg_loss, accuracy

# Main training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, '
          f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')

    scheduler.step()
```

### Pattern 2: Checkpointing

```python
def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

# Save best model
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss, 'best_model.pth')
```

---

## Summary

### Training Loop Essentials

```python
# The five-step mantra
optimizer.zero_grad()    # 1. Clear gradients
output = model(input)    # 2. Forward pass
loss = criterion(...)    # 3. Compute loss
loss.backward()          # 4. Backward pass
optimizer.step()         # 5. Update weights
```

### Key Decisions

| Decision | Recommendations |
|----------|----------------|
| **Loss** | CrossEntropy (classification), MSE (regression) |
| **Optimizer** | AdamW (general), SGD+momentum (vision) |
| **Learning Rate** | 1e-3 (Adam), 1e-1 (SGD), tune with LR finder |
| **Scheduler** | CosineAnnealing (general), ReduceLROnPlateau (adaptive) |
| **Gradient Clipping** | 1.0 (RNNs), 5.0 (Transformers), monitor first |

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Loss not decreasing | Check LR, verify loss function, check gradients |
| Loss → NaN | Lower LR, gradient clipping, check for log(0) or div by 0 |
| Slow convergence | Increase LR, better initialization, check data |
| Overfitting | Regularization (dropout, weight decay), more data |
| Underfitting | Bigger model, train longer, lower weight decay |

---

**Previous**: [03_modules.md](03_modules.md) - Building neural networks
**Next**: [Quick Reference](quick_reference.md) - One-page cheat sheet
