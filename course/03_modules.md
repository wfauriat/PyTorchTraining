# Modules: Building Neural Networks

> **Core Question**: How do we structure and organize neural network components in PyTorch?

**In this guide:**
- [Overview](#overview)
- [The nn.Module System](#the-nnmodule-system)
- [Parameters and Buffers](#parameters-and-buffers)
- [Submodules](#submodules-and-composition)
- [Forward Method](#the-forward-method)
- [Hooks](#hooks-for-inspection)
- [Initialization](#weight-initialization)
- [State Management](#state-management-traineval)
- [Serialization](#saving-and-loading-models)

---

## Overview

`nn.Module` is PyTorch's base class for all neural network components. It provides:
- **Parameter management**: Automatic tracking of learnable parameters
- **Hierarchy**: Composable modules within modules
- **State management**: Train/eval mode switching
- **Serialization**: Save and load model weights
- **GPU support**: Automatic device management

### The Module Hierarchy

```
Model (nn.Module)
├── Layer1 (nn.Module)
│   ├── weight (Parameter)
│   └── bias (Parameter)
├── Layer2 (nn.Module)
│   ├── weight (Parameter)
│   └── bias (Parameter)
└── activation (no parameters)
```

---

## The nn.Module System

### Basic Structure

```python
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()  # Always call parent __init__
        # Define components here

    def forward(self, x):
        # Define computation here
        return output
```

### Simple Example: Linear Regression

```python
class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(10, 1)
x = torch.randn(32, 10)
y = model(x)  # Calls forward()
```

**Key point**: Always call `model(x)`, not `model.forward(x)`. This ensures hooks and other machinery run correctly.

### Multi-Layer Network

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

---

## Parameters and Buffers

### Parameters

**Parameters** are tensors that should be learned during training.

```python
# Manually creating a parameter
self.weight = nn.Parameter(torch.randn(10, 5))

# Most layers create parameters automatically
self.linear = nn.Linear(10, 5)  # Creates weight and bias parameters
```

**Accessing parameters**:
```python
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Or just get tensors
params = list(model.parameters())
```

**Parameter groups** (for different learning rates):
```python
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-4},
    {'params': model.decoder.parameters(), 'lr': 1e-3}
])
```

### Buffers

**Buffers** are tensors that should be saved with the model but not trained (e.g., running statistics in BatchNorm).

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Register a buffer
        self.register_buffer('running_mean', torch.zeros(10))

    def forward(self, x):
        # Update running statistics (no gradients needed)
        self.running_mean = 0.9 * self.running_mean + 0.1 * x.mean(dim=0)
        return x - self.running_mean
```

**Key differences**:

| Aspect | Parameter | Buffer |
|--------|-----------|--------|
| Requires grad | Yes | No |
| Updated by optimizer | Yes | No |
| Saved in state_dict | Yes | Yes |
| Moved with `.to(device)` | Yes | Yes |
| Example | Weights, biases | BatchNorm running stats |

---

## Submodules and Composition

### ModuleList

For a **list** of modules:

```python
class DynamicNet(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
```

**Why not a Python list?**
```python
# ❌ Don't do this
self.layers = [nn.Linear(10, 10) for _ in range(5)]
# Parameters won't be registered!

# ✅ Do this
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
```

### ModuleDict

For a **dictionary** of modules:

```python
class MultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({
            'image': nn.Conv2d(3, 64, 3),
            'text': nn.Embedding(1000, 64)
        })

    def forward(self, image=None, text=None):
        outputs = {}
        if image is not None:
            outputs['image'] = self.encoders['image'](image)
        if text is not None:
            outputs['text'] = self.encoders['text'](text)
        return outputs
```

### Sequential

For **sequential** operations (no branching):

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Equivalent to:
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

**With OrderedDict** (named layers):
```python
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10))
]))

# Access by name
model.fc1.weight
```

### Nested Modules

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Residual connection
        return F.relu(x)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 7, padding=3)
        self.blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(5)
        ])
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        return self.fc(x)
```

---

## The Forward Method

### Rules for forward()

1. **Must return a value** (tensor or tuple of tensors)
2. **No need to override backward()** (autograd handles it)
3. **Can be arbitrarily complex** (loops, conditionals, etc.)

### Dynamic Computation

```python
class DynamicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x, num_repeats):
        # Different computation graph each call!
        for _ in range(num_repeats):
            x = F.relu(self.fc(x))
        return x

model = DynamicNet()
y1 = model(x, num_repeats=3)  # 3 layers
y2 = model(x, num_repeats=5)  # 5 layers (different graph)
```

### Multiple Outputs

```python
class MultiOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 64)
        self.decoder1 = nn.Linear(64, 10)
        self.decoder2 = nn.Linear(64, 5)

    def forward(self, x):
        encoded = self.encoder(x)
        out1 = self.decoder1(encoded)
        out2 = self.decoder2(encoded)
        return out1, out2

model = MultiOutput()
output1, output2 = model(x)
```

---

## Hooks for Inspection

Hooks allow you to inspect or modify intermediate values during forward/backward passes.

### Forward Hooks

```python
def forward_hook(module, input, output):
    print(f"Module: {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# Register hook on second layer
handle = model[2].register_forward_hook(forward_hook)

x = torch.randn(3, 10)
model(x)  # Hook will print info

handle.remove()  # Remove hook when done
```

### Backward Hooks

```python
def backward_hook(module, grad_input, grad_output):
    print(f"Gradient w.r.t. output: {grad_output[0].norm()}")
    # Can modify gradient here
    return grad_input  # Return modified gradient or None

handle = model[2].register_full_backward_hook(backward_hook)

y = model(x)
y.sum().backward()

handle.remove()
```

### Use Cases

- **Feature extraction**: Save activations from intermediate layers
- **Gradient inspection**: Monitor gradient flow
- **Gradient modification**: Implement custom gradient clipping
- **Visualization**: Extract activations for visualization

---

## Weight Initialization

### Why Initialization Matters

Poor initialization → vanishing/exploding gradients → slow or failed training

### Common Strategies

```python
import torch.nn.init as init

# Xavier/Glorot initialization (for sigmoid/tanh)
init.xavier_uniform_(layer.weight)
init.xavier_normal_(layer.weight)

# He/Kaiming initialization (for ReLU)
init.kaiming_uniform_(layer.weight, nonlinearity='relu')
init.kaiming_normal_(layer.weight, nonlinearity='relu')

# Constant initialization
init.constant_(layer.bias, 0)

# Orthogonal initialization (for RNNs)
init.orthogonal_(layer.weight)
```

### Initialize All Layers

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

model = MLP(10, 64, 10)
model.apply(init_weights)  # Recursively apply to all submodules
```

### Default Initializations

PyTorch layers have sensible defaults:

| Layer | Weight Init | Bias Init |
|-------|-------------|-----------|
| `nn.Linear` | Uniform $[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}]$ | Same |
| `nn.Conv2d` | Kaiming uniform | Zeros |
| `nn.LSTM` | Orthogonal | Zeros |
| `nn.Embedding` | Normal(0, 1) | N/A |

---

## State Management: train/eval

### Train vs Eval Mode

Some layers behave differently during training vs inference:

| Layer | Training | Evaluation |
|-------|----------|------------|
| **Dropout** | Randomly zeros units | Pass-through (no dropout) |
| **BatchNorm** | Update running stats | Use frozen stats |

### Switching Modes

```python
model.train()      # Set to training mode
# ... training loop ...

model.eval()       # Set to evaluation mode
with torch.no_grad():
    predictions = model(test_data)
```

### Checking Current Mode

```python
model.training  # Boolean: True if in training mode

# Check specific modules
for module in model.modules():
    if isinstance(module, nn.Dropout):
        print(f"Dropout training: {module.training}")
```

### Example: Dropout Behavior

```python
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.Dropout(p=0.5),  # 50% dropout
    nn.Linear(10, 1)
)

x = torch.randn(32, 10)

model.train()
y1 = model(x)  # Random units dropped

model.eval()
y2 = model(x)  # No dropout, all units active
```

---

## Saving and Loading Models

### State Dict

The `state_dict` contains all parameters and buffers:

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Inspecting State Dict

```python
state = model.state_dict()
for key, value in state.items():
    print(f"{key}: {value.shape}")
```

### Entire Model vs State Dict

```python
# Option 1: Save entire model (includes architecture)
torch.save(model, 'entire_model.pth')
model = torch.load('entire_model.pth')

# Option 2: Save only state dict (recommended)
torch.save(model.state_dict(), 'state.pth')
model = MyModel()  # Need to recreate architecture
model.load_state_dict(torch.load('state.pth'))
```

**Recommendation**: Save state dict only. It's more flexible and portable.

### Saving Optimizer State

```python
# Save both model and optimizer
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

### Partial Loading (Transfer Learning)

```python
pretrained_dict = torch.load('pretrained.pth')
model_dict = model.state_dict()

# Filter out keys we don't want
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and v.shape == model_dict[k].shape}

# Update with pretrained weights
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```

### Strict Loading

```python
# By default, strict=True (all keys must match)
model.load_state_dict(state_dict)

# Allow missing/unexpected keys
model.load_state_dict(state_dict, strict=False)
```

---

## Common Patterns

### Pattern 1: Freezing Layers

```python
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the final layer
for param in model.fc.parameters():
    param.requires_grad = True

# Or freeze specific layers
model.encoder.requires_grad_(False)  # Recursively set
model.decoder.requires_grad_(True)
```

### Pattern 2: Feature Extractor

```python
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        # Remove final classification layer
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.features.requires_grad_(False)  # Freeze

    def forward(self, x):
        return self.features(x)
```

### Pattern 3: Custom Layer

```python
class MyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Custom computation
        return F.linear(x, self.weight, self.bias) ** 2  # Squared linear

    def extra_repr(self):
        # For nice printing
        return f"in_features={self.weight.size(1)}, out_features={self.weight.size(0)}"
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **nn.Module** | Base class for all neural network components |
| **Parameters** | Learnable weights (tracked by optimizers) |
| **Buffers** | Non-learnable state (e.g., BatchNorm stats) |
| **Submodules** | Composable modules (ModuleList, ModuleDict, Sequential) |
| **Hooks** | Inspect/modify activations and gradients |
| **train/eval** | Switch behavior for layers like Dropout, BatchNorm |
| **state_dict** | Dictionary of all parameters and buffers |

### Module Checklist

When creating a custom module:
- ✓ Call `super().__init__()`
- ✓ Use `nn.Parameter` for learnable weights
- ✓ Use `register_buffer` for non-learnable state
- ✓ Use `ModuleList`/`ModuleDict` for sub-modules
- ✓ Call model via `model(x)`, not `model.forward(x)`
- ✓ Set `train()`/`eval()` mode appropriately
- ✓ Initialize weights if needed

### Common Mistakes

❌ Using Python list instead of ModuleList:
```python
self.layers = [nn.Linear(10, 10) for _ in range(5)]  # Parameters not registered!
```

✅ Use ModuleList:
```python
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
```

❌ Forgetting to switch to eval mode:
```python
model(test_data)  # Dropout/BatchNorm still in training mode!
```

✅ Explicitly set eval mode:
```python
model.eval()
with torch.no_grad():
    model(test_data)
```

---

**Previous**: [02_autograd.md](02_autograd.md) - Automatic differentiation
**Next**: [04_training.md](04_training.md) - Training loop mechanics

**Related**:
- [Quick Reference](quick_reference.md) - Module API cheat sheet
