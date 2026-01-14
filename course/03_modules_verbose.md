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

The **`nn.Module`** class is the foundational building block of all neural networks in PyTorch. It serves as the base class from which all neural network components inherit, providing a sophisticated system for managing the complex hierarchies of parameters, sub-components, and behaviors that modern neural networks require. Understanding `nn.Module` deeply is essential because it's not just a convenience wrapper—it's the architectural pattern that makes PyTorch's approach to neural networks both powerful and elegant.

When you create a custom layer, a complete model, or even just a single operation that needs to interact with the rest of PyTorch's infrastructure, you inherit from `nn.Module`. This inheritance automatically provides you with a rich set of capabilities that would be tedious and error-prone to implement manually:

**Parameter management**: The module system automatically tracks all learnable parameters (weights and biases) recursively through your entire model hierarchy. When you nest one module inside another, the parent automatically discovers and tracks the child's parameters. This means you can build arbitrarily complex architectures through composition, and PyTorch will still be able to collect all the parameters that need to be optimized.

**Hierarchical composition**: Modules can contain other modules, which can contain yet more modules, creating tree structures that mirror the conceptual organization of your network. This composability is crucial for building modern architectures where a "ResNet block" might contain multiple convolutional layers, batch normalization layers, and residual connections, and then you might stack dozens of these blocks together.

**State management**: Neural networks need to behave differently during training versus evaluation—dropout layers must be disabled during inference, batch normalization must use frozen statistics rather than updating them, and so on. The module system provides `.train()` and `.eval()` methods that propagate these mode changes recursively through the entire model hierarchy automatically.

**Serialization and deserialization**: The module system provides `.state_dict()` to extract all parameters and buffers (persistent non-parameter state like running statistics) into a dictionary, and `.load_state_dict()` to restore them. This enables model checkpointing, resuming training, transfer learning, and deployment.

**Device management**: When you call `.to(device)` on a module, it automatically moves all parameters and buffers to the specified device (CPU or GPU), recursively through the entire hierarchy. This makes managing GPU computation simple and natural.

### The Module Hierarchy: Understanding the Tree Structure

A typical neural network forms a tree-like hierarchy that mirrors its logical structure:

```
Model (nn.Module)
├── Layer1 (nn.Module)
│   ├── weight (Parameter)
│   └── bias (Parameter)
├── Layer2 (nn.Module)
│   ├── weight (Parameter)
│   └── bias (Parameter)
└── activation (nn.Module, no parameters)
```

Each node in this tree is a module, and the leaf nodes are either parameters (learnable tensors) or buffers (persistent non-learnable tensors). When you call `model.parameters()`, PyTorch traverses this tree and collects all the parameter leaves. When you call `model.to('cuda')`, it traverses the tree and moves every tensor it finds. This recursive, compositional design is what makes PyTorch's module system so powerful and expressive.

---

## The nn.Module System

### Basic Structure: The Template Every Module Follows

Every custom module follows a consistent pattern that has become second nature to PyTorch practitioners:

```python
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()  # ALWAYS call parent __init__ first
        # Define components here (submodules, parameters, buffers)

    def forward(self, x):
        # Define the computation here
        return output
```

The **`__init__` method** is where you define the structure of your module—the submodules it contains, the parameters it owns, and any buffers or configuration it needs. Crucially, you must always call `super().__init__()` at the very beginning. This initializes PyTorch's parameter tracking machinery; without it, your module won't properly register parameters or submodules.

The **`forward` method** is where you define the actual computation your module performs. This method takes input tensors and returns output tensors. You don't manually call `forward()`—instead, you treat the module like a function and call `model(x)`, which invokes `forward(x)` behind the scenes along with other important machinery like hooks (which we'll cover later). This distinction matters because calling `model(x)` triggers the module's `__call__` method, which does important setup before and after calling `forward()`.

### Simple Example: Linear Regression

Let's start with the simplest possible example—a linear regression model that's just a single linear layer:

```python
class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # nn.Linear creates weight and bias parameters automatically
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Forward pass: just apply the linear transformation
        return self.linear(x)

# Create an instance
model = LinearRegression(in_features=10, out_features=1)

# Use it
x = torch.randn(32, 10)  # Batch of 32 examples, each with 10 features
y = model(x)  # Calls forward(), returns predictions of shape (32, 1)
```

**Important**: Notice that we call `model(x)`, not `model.forward(x)`. While both will execute the forward pass computation, calling `model(x)` is the correct idiom because it triggers the module's `__call__` method, which handles hooks, validation, and other internal machinery. Always use `model(x)` in practice.

When we created `self.linear = nn.Linear(in_features, out_features)`, several things happened automatically:
1. PyTorch registered `self.linear` as a **submodule** of our `LinearRegression` module
2. The `nn.Linear` layer created its own weight matrix (shape: out_features × in_features) and bias vector (shape: out_features)
3. These parameters were automatically registered so that `model.parameters()` will find them

This automatic registration is why we assign the layer to `self.linear` rather than just creating it inside `forward()`. Assignments to `self` in `__init__` trigger PyTorch's tracking machinery.

### Multi-Layer Perceptron: Composing Multiple Layers

Most neural networks involve multiple layers composed together. Here's a simple multi-layer perceptron (MLP) with two hidden layers:

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Create three linear layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Activation function (ReLU doesn't have parameters, but we can store it)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply layers with activations in between
        x = self.relu(self.fc1(x))  # First hidden layer
        x = self.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)             # Output layer (no activation)
        return x

model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
```

This pattern—defining the structure in `__init__`, then composing the computation in `forward()`—is ubiquitous in PyTorch. Notice how clean the `forward()` method is: it reads like a straightforward description of the computation flow, without any parameter management clutter.

**Why store `self.relu`?** Technically, you could just call `F.relu(x)` directly in `forward()` since ReLU has no parameters. However, storing it as `self.relu = nn.ReLU()` makes the architecture more explicit and enables certain functionality like hooks (which we'll explore later). Both approaches work, and you'll see both in practice.

---

## Parameters and Buffers

Understanding the distinction between **parameters** and **buffers** is crucial for correctly implementing custom modules and for understanding how built-in modules like `BatchNorm` work.

### Parameters: Learnable Tensors

**Parameters** are tensors that should be learned during training through gradient descent. They're what the optimizer updates when you call `optimizer.step()`. Parameters have `requires_grad=True` by default, meaning gradients will be computed for them during backpropagation.

**Creating parameters manually:**

```python
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Manually create a parameter
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Use the parameters in computation
        return x @ self.weight.t() + self.bias
```

When you wrap a tensor with `nn.Parameter()`, you're telling PyTorch three things:
1. This tensor should be optimized (it needs gradients)
2. This tensor should be included when saving/loading the model
3. This tensor should be moved when you call `.to(device)`

**Most built-in layers create parameters automatically**: When you use `nn.Linear`, `nn.Conv2d`, `nn.Embedding`, etc., these layers create appropriately sized and initialized parameters for you. You rarely need to create parameters manually unless you're implementing a novel layer type.

**Accessing all parameters in a model:**

```python
# Iterate over all parameters with their names
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Just get the parameter tensors (no names)
params = list(model.parameters())

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
```

**Parameter groups with different learning rates:**

Sometimes you want to optimize different parts of your model with different learning rates (common in transfer learning):

```python
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-4},
    {'params': model.decoder.parameters(), 'lr': 1e-3}
])
```

This creates an optimizer that will update the encoder's parameters with learning rate 1e-4 and the decoder's parameters with learning rate 1e-3.

### Buffers: Persistent Non-Learnable State

**Buffers** are tensors that should be saved with the model but **not** trained by the optimizer. They're used for persistent state that needs to be maintained across forward passes but isn't learned through gradient descent. The canonical example is batch normalization's running statistics.

**Creating buffers:**

```python
class MyNormalization(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Register a buffer for running mean
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # During training: update running statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)

            # Exponential moving average update
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var

            # Normalize using batch statistics
            return (x - batch_mean) / (batch_var + 1e-5).sqrt()
        else:
            # During evaluation: use running statistics
            return (x - self.running_mean) / (self.running_var + 1e-5).sqrt()
```

**Key differences between parameters and buffers:**

| Aspect | Parameter | Buffer |
|--------|-----------|--------|
| Requires gradients | Yes (`requires_grad=True`) | No (`requires_grad=False`) |
| Updated by optimizer | Yes | No (you update manually) |
| Saved in `state_dict()` | Yes | Yes |
| Moved by `.to(device)` | Yes | Yes |
| Returned by `.parameters()` | Yes | No |
| Returned by `.buffers()` | No | Yes |
| **Typical examples** | Weights, biases, embeddings | Running stats (BatchNorm), positional encodings (Transformers) |

The key insight is that buffers are for state that needs to persist and be saved, but doesn't participate in gradient-based optimization. Without buffers, you'd have to manually manage saving and loading this state, manually move it between devices, and keep track of it separately—buffers handle all of this automatically.

---

## Submodules and Composition

Real neural networks are composed of many layers organized into hierarchical structures. PyTorch provides several containers for managing collections of submodules.

### ModuleList: For Lists of Modules

When you need to create a **dynamic number of modules** or store modules in a list, use **`nn.ModuleList`**:

```python
class DynamicNet(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        # Create a list of linear layers
        self.layers = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(num_layers)
        ])

    def forward(self, x):
        # Apply each layer with activation
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

# Can create networks with different depths
model_3layers = DynamicNet(num_layers=3)
model_10layers = DynamicNet(num_layers=10)
```

**Why not just use a Python list?** This is a crucial point:

```python
# ❌ WRONG: Parameters won't be registered!
self.layers = [nn.Linear(10, 10) for _ in range(5)]
# model.parameters() will NOT find these layers' parameters

# ✅ CORRECT: Use ModuleList
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
# model.parameters() WILL find all parameters
```

A regular Python list doesn't trigger PyTorch's module registration system. When you assign a plain list to `self.layers`, PyTorch doesn't know to look inside that list for submodules. `nn.ModuleList` is a special container that tells PyTorch "this is a list of modules, please register them all." This is essential for:
- Optimizer finding parameters: `optimizer = Adam(model.parameters())` needs to find all parameters
- Device movement: `model.to('cuda')` needs to move all parameters and buffers
- Saving and loading: `state_dict()` needs to save all parameters

### ModuleDict: For Dictionaries of Modules

When you need to organize modules with **string keys** (like different processing pathways for different input types), use **`nn.ModuleDict`**:

```python
class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Different encoders for different modalities
        self.encoders = nn.ModuleDict({
            'image': nn.Conv2d(3, 64, kernel_size=3),
            'text': nn.Embedding(vocab_size=10000, embedding_dim=64),
            'audio': nn.Linear(128, 64)
        })

        # Shared classifier
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs):
        # inputs is a dict like {'image': image_tensor, 'text': text_tensor}
        encoded = {}

        # Process each modality present in the input
        for modality, data in inputs.items():
            if modality in self.encoders:
                encoded[modality] = self.encoders[modality](data)

        # Combine encodings (e.g., average)
        combined = torch.stack(list(encoded.values())).mean(dim=0)
        return self.classifier(combined)
```

`ModuleDict` provides dictionary-like access with string keys while ensuring that all contained modules are properly registered. This is particularly useful for multi-task learning, multi-modal models, or any architecture where you need named, dynamically-selected processing pathways.

### Sequential: For Linear Pipelines

When your architecture is a simple **linear sequence** of operations with no branching or complex control flow, **`nn.Sequential`** provides the most concise syntax:

```python
# Define a model as a sequence
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

# Use it like any module
x = torch.randn(32, 784)
output = model(x)
```

`Sequential` automatically chains the operations together—the output of each layer becomes the input to the next. It's equivalent to writing:

```python
class SequentialEquivalent(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return x
```

But much more concise! The tradeoff is flexibility: `Sequential` only works for simple linear pipelines. If you need branching (like residual connections), multiple inputs, or complex control flow, you'll need to write a full `nn.Module` subclass with a custom `forward()` method.

**Named Sequential with OrderedDict:**

By default, layers in `Sequential` are accessed by index (`model[0]`, `model[1]`, etc.). For better readability, you can give them names:

```python
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(0.2)),
    ('fc2', nn.Linear(256, 10))
]))

# Access by name
print(model.fc1.weight.shape)

# Still works by index
print(model[0].weight.shape)
```

Named layers make debugging easier and make your saved models more interpretable.

### Nested Modules: Building Complex Architectures

The real power of PyTorch's module system emerges when you nest custom modules inside each other. Here's a simple ResNet-style architecture to illustrate:

```python
class ResidualBlock(nn.Module):
    """A single residual block with skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Save input for residual connection
        residual = x

        # Main path: conv -> batchnorm -> relu -> conv -> batchnorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # Add residual connection (skip connection)
        x = x + residual

        # Final activation
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    """Simple ResNet using our residual blocks."""
    def __init__(self, num_blocks=5, num_classes=10):
        super().__init__()
        # Initial convolution
        self.conv_in = nn.Conv2d(3, 64, kernel_size=7, padding=3)

        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(channels=64) for _ in range(num_blocks)
        ])

        # Classification head
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input: (batch, 3, height, width)
        x = self.conv_in(x)  # (batch, 64, height, width)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)  # Each block maintains shape

        # Global average pooling
        x = x.mean(dim=[2, 3])  # (batch, 64)

        # Classification
        x = self.fc(x)  # (batch, num_classes)
        return x
```

This architecture demonstrates several key patterns:
1. **Composability**: `ResidualBlock` is a complete module that we nest inside `ResNet`
2. **Automatic registration**: All parameters in the blocks are automatically discovered
3. **Clean separation**: Each block encapsulates its own structure and forward logic
4. **Skip connections**: Complex control flow (adding residual) is straightforward in `forward()`

When you call `model.parameters()` on this `ResNet`, PyTorch automatically traverses the entire hierarchy: it finds `conv_in`, iterates through all the blocks in `self.blocks`, and within each block finds `conv1`, `conv2`, `bn1`, and `bn2`, collecting all of their parameters. This recursive discovery is what makes compositional architectures so natural in PyTorch.

---

## The Forward Method

The **`forward()` method** is where the computation happens. It defines what your module does when it receives input. Understanding the rules and patterns for `forward()` is essential for effective PyTorch usage.

### Rules and Guidelines for forward()

1. **Must return a value**: The `forward()` method must return at least one tensor (or a tuple/dict of tensors). This return value becomes the output when you call `model(x)`.

2. **No need to implement backward()**: Unlike older frameworks or when implementing custom autograd functions, you never need to implement a backward pass for modules. PyTorch's autograd automatically computes gradients through whatever computation you write in `forward()`.

3. **Can be arbitrarily complex**: Your `forward()` method can use Python control flow (if statements, loops, recursion), call helper methods, use Python data structures—anything goes. The computation graph is built dynamically based on what actually executes.

4. **Can take multiple arguments**: While we usually write `forward(self, x)`, you can define any signature: `forward(self, x, lengths, mask)`, etc.

5. **Must be deterministic for the same inputs** (usually): For reproducibility and certain algorithms, `forward()` should generally produce the same output given the same input (when in `eval()` mode).

### Dynamic Computation: Exploiting PyTorch's Flexibility

One of PyTorch's key advantages is that the computation graph is built dynamically, which means your `forward()` method can do different things on different calls:

```python
class AdaptiveDepthNetwork(nn.Module):
    """Network whose depth changes based on input properties."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x, num_repeats):
        # Different computation graph for different num_repeats!
        for _ in range(num_repeats):
            x = F.relu(self.fc(x))
        return x

model = AdaptiveDepthNetwork()

# 3 applications of the layer (shallow network)
y1 = model(x, num_repeats=3)

# 10 applications of the layer (deep network)
y2 = model(x, num_repeats=10)
```

Each call builds a different computation graph—the graph for `num_repeats=3` has 3 linear layers and 3 ReLU activations, while the graph for `num_repeats=10` has 10 of each. This would be impossible or extremely awkward in a static-graph framework.

**Real-world uses for dynamic computation:**
- **Variable-length sequences**: Processing sequences of different lengths without padding to maximum length
- **Adaptive computation time**: Deciding how many layers to apply based on input difficulty
- **Tree-structured networks**: Processing tree or graph structures where the computation depends on the structure
- **Neural architecture search**: Networks that dynamically modify their own structure

### Multiple Outputs: Returning Structured Data

Your `forward()` method can return tuples, dicts, or custom objects when you need multiple outputs:

```python
class MultiHeadModel(nn.Module):
    """Model with multiple prediction heads."""
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Two separate prediction heads
        self.classification_head = nn.Linear(64, 10)
        self.regression_head = nn.Linear(64, 1)

    def forward(self, x):
        # Shared feature extraction
        features = self.shared_encoder(x)

        # Two separate predictions
        class_logits = self.classification_head(features)
        regression_output = self.regression_head(features)

        # Return both as a tuple
        return class_logits, regression_output

model = MultiHeadModel()
x = torch.randn(32, 10)

# Unpack the outputs
class_predictions, regression_predictions = model(x)

# Compute separate losses
classification_loss = F.cross_entropy(class_predictions, class_targets)
regression_loss = F.mse_loss(regression_predictions, regression_targets)

# Combined loss
total_loss = classification_loss + regression_loss
```

This pattern is common in multi-task learning where a single model needs to solve multiple related problems simultaneously.

---

## Hooks for Inspection

**Hooks** are a powerful but often underutilized feature that allow you to insert custom code at specific points during forward or backward passes without modifying the module itself. They're essential for debugging, visualization, and implementing certain advanced techniques.

### Forward Hooks: Inspecting Activations

A **forward hook** is a function that gets called after a module's `forward()` method completes. It receives the module, its input, and its output:

```python
def activation_hook(module, input, output):
    """
    Called after module's forward pass.

    Args:
        module: The module that executed
        input: Tuple of input tensors to the module
        output: The output tensor from the module
    """
    print(f"Module: {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print()

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Register hook on the first linear layer
handle = model[0].register_forward_hook(activation_hook)

# Run forward pass
x = torch.randn(3, 10)
output = model(x)  # Hook will be called and print info

# Remove hook when done
handle.remove()
```

**Important note**: The `input` parameter is a tuple even if there's only one input tensor. Access it as `input[0]`.

**Common use cases for forward hooks:**

1. **Feature extraction**: Save intermediate activations for visualization or analysis
2. **Debugging**: Inspect shapes and values at specific layers to diagnose issues
3. **Monitoring**: Track activation statistics during training
4. **Attention visualization**: Extract attention weights in Transformer models

**Example: Collecting features from multiple layers:**

```python
class FeatureExtractor:
    def __init__(self, model, layer_names):
        self.features = {}
        self.hooks = []

        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self.save_features(name))
                self.hooks.append(hook)

    def save_features(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def remove(self):
        for hook in self.hooks:
            hook.remove()

# Usage
model = MyComplexModel()
extractor = FeatureExtractor(model, ['encoder.layer3', 'decoder.layer1'])

output = model(input)
features = extractor.features  # Dict of extracted features

extractor.remove()  # Clean up
```

### Backward Hooks: Inspecting Gradients

A **backward hook** is called during the backward pass, receiving gradients:

```python
def gradient_hook(module, grad_input, grad_output):
    """
    Called during backward pass.

    Args:
        module: The module
        grad_input: Tuple of gradients w.r.t. the module's inputs
        grad_output: Tuple of gradients w.r.t. the module's outputs
    """
    print(f"Gradient w.r.t. output: {grad_output[0].norm().item():.4f}")
    # Can modify and return new grad_input to change backpropagation
    return grad_input

handle = model[0].register_full_backward_hook(gradient_hook)

output = model(input)
loss = output.sum()
loss.backward()  # Hook will be called during backprop

handle.remove()
```

**Use cases for backward hooks:**
- **Gradient clipping per-layer**: Apply different clipping thresholds to different layers
- **Debugging vanishing/exploding gradients**: Monitor gradient magnitudes through the network
- **Custom gradient modification**: Implement techniques like gradient reversal layers
- **Research and analysis**: Study how gradients flow through different architectures

### A Word of Caution: Hooks and Performance

Hooks have a cost—they add function call overhead and can prevent certain optimizations. For production inference, remove all hooks. They're primarily development and research tools.

---

## Weight Initialization

The way you initialize your neural network's weights has a profound impact on whether training converges successfully, and if so, how quickly. Poor initialization can lead to vanishing or exploding gradients, effectively preventing learning.

### Why Initialization Matters: The Gradient Flow Problem

Consider a deep network with many layers. During backpropagation, gradients flow backward through each layer via the chain rule, which multiplies gradients together. If the weights are initialized such that gradients tend to shrink at each layer, by the time gradients reach early layers they'll be vanishingly small, and those layers won't learn. Conversely, if gradients tend to grow at each layer, they'll explode to infinity.

The mathematical insight behind modern initialization schemes is to choose the scale of initial weights such that activations and gradients maintain approximately constant variance as they flow through the network. This keeps the network in a regime where gradients are informative without being extreme.

### Common Initialization Strategies

PyTorch provides several standard initialization schemes in the `torch.nn.init` module:

```python
import torch.nn.init as init

# Xavier/Glorot initialization (designed for sigmoid/tanh activations)
init.xavier_uniform_(layer.weight)  # Uniform distribution
init.xavier_normal_(layer.weight)   # Normal distribution

# He/Kaiming initialization (designed for ReLU activations)
init.kaiming_uniform_(layer.weight, nonlinearity='relu')
init.kaiming_normal_(layer.weight, nonlinearity='relu')

# Orthogonal initialization (useful for RNN weights)
init.orthogonal_(layer.weight)

# Simple constant initialization
init.constant_(layer.bias, 0)  # Common to initialize biases to zero
init.constant_(layer.weight, 0.01)

# Uniform initialization in a range
init.uniform_(layer.weight, a=-0.1, b=0.1)

# Normal initialization with specific mean and std
init.normal_(layer.weight, mean=0, std=0.02)
```

**Xavier/Glorot initialization** maintains variance when using sigmoid or tanh activations by sampling weights from a distribution with variance `2 / (fan_in + fan_out)`, where `fan_in` is the number of input units and `fan_out` is the number of output units.

**He/Kaiming initialization** is similar but designed for ReLU activations, which kill half the activations on average. It uses variance `2 / fan_in` to compensate.

**Orthogonal initialization** initializes weights to an orthogonal matrix, which can help with gradient flow in recurrent networks by preventing gradients from growing or shrinking exponentially.

### Initializing All Layers in a Model

Rather than manually initializing each layer, you can recursively apply an initialization function to all submodules:

```python
def init_weights(module):
    """
    Initialize weights for different layer types.
    This function will be applied recursively to all submodules.
    """
    if isinstance(module, nn.Linear):
        # Linear layers: use Kaiming initialization for weights
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        # Biases: initialize to zero
        if module.bias is not None:
            init.constant_(module.bias, 0)

    elif isinstance(module, nn.Conv2d):
        # Convolutional layers: Kaiming initialization
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)

    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        # Normalization layers: scale to 1, shift to 0
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)

# Apply to entire model
model = MyComplexModel()
model.apply(init_weights)
```

The `.apply()` method recursively applies the given function to every submodule in the hierarchy, making it easy to initialize complex models with a single call.

### Default Initializations in PyTorch Layers

PyTorch's built-in layers come with sensible default initializations, so you often don't need to initialize manually:

| Layer Type | Default Weight Initialization | Default Bias Initialization |
|------------|------------------------------|---------------------------|
| `nn.Linear` | Uniform $[-\sqrt{k}, \sqrt{k}]$ where $k = 1/\text{in\_features}$ | Same as weights |
| `nn.Conv2d` | Kaiming uniform with ReLU nonlinearity | Uniform like Linear |
| `nn.LSTM` | Orthogonal for recurrent weights, Xavier for input weights | Zeros |
| `nn.Embedding` | Normal(0, 1) | N/A |
| `nn.BatchNorm2d` | Weight=1, Bias=0 (for affine=True) | N/A |

These defaults are generally good, but you might want custom initialization for transfer learning, specific research techniques, or when you've found that a particular initialization works better for your task.

---

## State Management: train/eval

Neural networks often need to behave differently during training versus inference. PyTorch provides the `.train()` and `.eval()` methods to control this behavior, which automatically propagates through your entire model hierarchy.

### Layers That Change Behavior: Dropout and BatchNorm

Some layers behave fundamentally differently depending on whether you're training or evaluating:

**Dropout** randomly zeros a fraction of activations during training (to prevent overfitting), but during evaluation it must act as an identity function (pass everything through) so that predictions are deterministic:

```python
dropout = nn.Dropout(p=0.5)  # Drop 50% of activations during training

# Training mode
dropout.train()
x = torch.randn(10)
y_train = dropout(x)  # Some elements randomly zeroed

# Evaluation mode
dropout.eval()
y_eval = dropout(x)  # All elements pass through unchanged (y_eval == x)
```

**BatchNorm** computes mean and variance over the current batch during training and updates running statistics. During evaluation, it uses frozen running statistics (accumulated during training) instead of the current batch's statistics:

```python
bn = nn.BatchNorm1d(10)

# Training mode
bn.train()
y = bn(x)  # Uses batch statistics, updates running statistics

# Evaluation mode
bn.eval()
y = bn(x)  # Uses frozen running statistics, doesn't update them
```

This distinction is crucial: if you forget to call `.eval()` during inference, your batch normalization layers will behave incorrectly (computing statistics on small eval batches rather than using the trained running statistics), and dropout will randomly zero activations that should be active.

### Switching Modes: The Standard Pattern

The standard idiom is:

```python
# Training
model.train()  # Set model and all submodules to training mode
for batch in train_loader:
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()  # Set model and all submodules to evaluation mode
with torch.no_grad():  # Disable gradient computation for efficiency
    for batch in test_loader:
        output = model(batch)
        # Compute metrics, etc.
```

Note that `.train()` and `.eval()` recursively affect all submodules—you only need to call it on the top-level model.

### Checking Current Mode

You can query whether a module is in training mode:

```python
if model.training:
    print("Model is in training mode")
else:
    print("Model is in evaluation mode")

# Check specific submodules
for name, module in model.named_modules():
    if isinstance(module, (nn.Dropout, nn.BatchNorm2d)):
        print(f"{name}: training={module.training}")
```

### Custom Train/Eval Behavior

If you're implementing a custom module that needs different training and eval behavior, you can check `self.training` in your `forward()` method:

```python
class CustomModule(nn.Module):
    def forward(self, x):
        if self.training:
            # Training-specific behavior
            x = x + torch.randn_like(x) * 0.1  # Add noise during training
        else:
            # Evaluation-specific behavior
            x = x  # No noise during evaluation
        return x
```

---

## Saving and Loading Models

PyTorch provides flexible mechanisms for serializing and deserializing models, enabling checkpointing during training, resuming interrupted training, transfer learning, and model deployment.

### The State Dictionary: The Core Serialization Mechanism

The **state_dict** is a Python dictionary that maps parameter names to their tensor values. It contains all trainable parameters and registered buffers:

```python
# Get the state dictionary
state_dict = model.state_dict()

# It's just a dict mapping names to tensors
for param_name, param_tensor in state_dict.items():
    print(f"{param_name}: {param_tensor.shape}")

# Example output:
# encoder.weight: torch.Size([64, 10])
# encoder.bias: torch.Size([64])
# decoder.weight: torch.Size([10, 64])
# decoder.bias: torch.Size([10])
```

**Saving a model:**

```python
# Save the state dict to a file
torch.save(model.state_dict(), 'model_checkpoint.pth')
```

**Loading a model:**

```python
# Create model architecture (must match the saved model)
model = MyModel()

# Load the saved parameters
model.load_state_dict(torch.load('model_checkpoint.pth'))

# Set to evaluation mode for inference
model.eval()
```

This approach saves only the parameters, not the model architecture itself. This is the **recommended approach** because:
- It's more portable (not tied to the specific Python code that defined the model)
- It's more flexible (you can load into slightly modified architectures)
- It's safer (no arbitrary code execution from pickle)

### Saving the Entire Model (Less Recommended)

You can also save the entire model object:

```python
# Save entire model (architecture + parameters)
torch.save(model, 'entire_model.pth')

# Load entire model
model = torch.load('entire_model.pth')
```

This is simpler but has significant drawbacks:
- Less portable (requires the exact same class definitions to be available)
- Can break if you refactor your code
- Security risk (pickle can execute arbitrary code)

Use this only for quick experiments where you control both saving and loading contexts.

### Saving Training State: Checkpoints

When training large models, you want to save not just the model but the entire training state so you can resume if interrupted:

```python
# Save complete checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'learning_rate': current_lr,
    'best_val_accuracy': best_val_acc
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')

# Load complete checkpoint
checkpoint = torch.load('checkpoint_epoch_10.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
loss = checkpoint['loss']
```

This allows you to resume training exactly where you left off, including optimizer state (momentum buffers, adaptive learning rates, etc.).

### Partial Loading and Transfer Learning

Often you want to load some parameters from a pretrained model but not all (for example, when the final layer has a different number of classes):

```python
# Load pretrained parameters
pretrained_dict = torch.load('pretrained_model.pth')

# Get current model's state dict
model_dict = model.state_dict()

# Filter out keys that don't match in name or shape
pretrained_dict = {
    k: v for k, v in pretrained_dict.items()
    if k in model_dict and v.shape == model_dict[k].shape
}

# Update current model's state dict with pretrained values
model_dict.update(pretrained_dict)

# Load the updated state dict
model.load_state_dict(model_dict)
```

This pattern is common in transfer learning where you load pretrained feature extractors but initialize new classification heads.

### Strict Loading

By default, `load_state_dict()` requires that the state dict exactly matches the model's parameters (strict=True):

```python
# Strict loading (default): all keys must match exactly
model.load_state_dict(state_dict)  # Will error if any mismatch

# Non-strict loading: allows missing or unexpected keys
model.load_state_dict(state_dict, strict=False)
```

With `strict=False`, you can load partial models, but you won't get errors if you accidentally load the wrong checkpoint. Use strict loading when possible and non-strict loading only when you intentionally want partial loading.

---

## Common Patterns

### Pattern 1: Freezing and Unfreezing Layers

In transfer learning, you often want to freeze pretrained layers and only train new layers:

```python
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the final classification layer
for param in model.classifier.parameters():
    param.requires_grad = True

# Now only classifier parameters will be updated by optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
```

**Alternative using `.requires_grad_()`:**

```python
# Freeze encoder
model.encoder.requires_grad_(False)

# Unfreeze decoder
model.decoder.requires_grad_(True)
```

The `.requires_grad_()` method recursively sets `requires_grad` for all parameters in a module.

### Pattern 2: Feature Extractor

Extracting intermediate features from a pretrained model:

```python
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        # Remove the final classification layer(s)
        # children() returns immediate children, we exclude the last one
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])

        # Freeze parameters
        self.features.requires_grad_(False)

    def forward(self, x):
        return self.features(x)

# Usage
pretrained = torchvision.models.resnet50(pretrained=True)
extractor = FeatureExtractor(pretrained)

# Get features instead of classifications
features = extractor(images)
```

### Pattern 3: Custom Layer with Extra Representation

When implementing custom layers, overriding `extra_repr()` makes your model prints more informative:

```python
class SquaredLinear(nn.Module):
    """Custom layer that applies squared linear transformation."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Custom computation: squared linear transformation
        return (F.linear(x, self.weight, self.bias)) ** 2

    def extra_repr(self):
        # Custom string representation for print()
        return f'in_features={self.weight.size(1)}, out_features={self.weight.size(0)}'

# When you print the model, you'll see:
# SquaredLinear(in_features=10, out_features=5)
```

---

## Summary

### Key Concepts Synthesis

| Concept | Core Understanding |
|---------|-------------------|
| **nn.Module** | Base class for all neural network components. Provides parameter management, hierarchical composition, state management, serialization, and device management through automatic recursive traversal. |
| **Parameters** | Learnable tensors (`requires_grad=True`) that are updated by optimizers. Created with `nn.Parameter()` or automatically by built-in layers. |
| **Buffers** | Non-learnable persistent state (`requires_grad=False`) that is saved with the model but not optimized. Used for running statistics, positional encodings, etc. Registered with `register_buffer()`. |
| **Submodule Containers** | `ModuleList` for lists, `ModuleDict` for dicts, `Sequential` for linear pipelines. Essential for proper parameter registration—never use plain Python lists for modules. |
| **forward() Method** | Defines the module's computation. Can be arbitrarily complex with Python control flow. Called automatically via `model(x)`, not `model.forward(x)`. |
| **Hooks** | Callbacks inserted into forward/backward passes for inspection and modification. Used for debugging, visualization, and advanced techniques. |
| **Initialization** | Critical for gradient flow in deep networks. Use Xavier for sigmoid/tanh, Kaiming for ReLU, orthogonal for RNNs. Apply recursively with `model.apply()`. |
| **train/eval Modes** | Control behavior of layers like Dropout and BatchNorm. Always call `model.eval()` before inference and `model.train()` before training. |
| **state_dict** | Dictionary mapping parameter names to tensors. The recommended serialization format. Save with `torch.save(model.state_dict(), path)`. |

### The Module Lifecycle Pattern

```python
# 1. Define architecture
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(...)

    def forward(self, x):
        return self.layers(x)

# 2. Instantiate and initialize
model = MyModel()
model.apply(init_weights)  # Custom initialization if needed

# 3. Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 4. Training loop
model.train()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(num_epochs):
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'checkpoint_epoch_{epoch}.pth')

# 5. Evaluation
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        predictions = model(batch)
        # Evaluate...

# 6. Save final model
torch.save(model.state_dict(), 'final_model.pth')
```

### Critical Checklist for Custom Modules

When implementing your own `nn.Module`, ensure:

- ✓ **Call `super().__init__()`** first thing in `__init__`
- ✓ **Use `nn.Parameter`** for learnable weights
- ✓ **Use `register_buffer`** for persistent non-learnable state
- ✓ **Use `ModuleList`/`ModuleDict`** for collections of submodules, never plain Python lists/dicts
- ✓ **Call via `model(x)`** not `model.forward(x)` (hooks won't run otherwise)
- ✓ **Set `.train()`/`.eval()`** appropriately for training vs inference
- ✓ **Initialize weights** if defaults aren't suitable
- ✓ **Implement `extra_repr()`** for better debugging (optional but helpful)

### Common Mistakes to Avoid

**❌ Using Python list for modules:**
```python
self.layers = [nn.Linear(10, 10) for _ in range(5)]  # Parameters lost!
```

**✅ Use ModuleList:**
```python
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
```

**❌ Forgetting eval mode during inference:**
```python
predictions = model(test_data)  # Dropout/BatchNorm misbehave!
```

**✅ Explicitly set eval mode:**
```python
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

**❌ Calling forward directly:**
```python
output = model.forward(input)  # Hooks don't run!
```

**✅ Call the module as a function:**
```python
output = model(input)
```

---

**Previous**: [02_autograd.md](02_autograd.md) - Understanding automatic differentiation and computation graphs
**Next**: [04_training.md](04_training.md) - The training loop, optimizers, and learning strategies

**Related**:
- [Quick Reference](quick_reference.md) - Module API cheat sheet
- PyTorch nn.Module documentation for additional advanced features
