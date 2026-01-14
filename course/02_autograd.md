# Automatic Differentiation with Autograd

> **Core Question**: How does PyTorch compute gradients automatically for backpropagation?

**In this guide:**
- [Overview](#overview)
- [Computation Graphs](#computation-graphs)
- [The Backward Pass](#the-backward-pass)
- [Gradient Flow](#gradient-flow-and-accumulation)
- [Context Managers](#context-managers)
- [Custom Autograd](#custom-autograd-functions)
- [Debugging](#debugging-gradients)
- [Advanced Topics](#advanced-topics)

---

## Overview

**Automatic differentiation (autograd)** is PyTorch's system for computing derivatives of tensor operations. It's the engine that powers neural network training through backpropagation.

### Why Autograd Matters

Without autograd, you'd need to:
1. Manually derive gradient formulas for every operation
2. Implement backward passes by hand
3. Debug complex derivative chains

With autograd, PyTorch:
- **Records** operations as you compute
- **Builds** a computation graph
- **Computes** gradients automatically with `.backward()`

### The Big Picture

```
Forward pass:  Input → Operations → Output
               (PyTorch records this)

Backward pass: Output ← Gradients ← Input
               (.backward() computes this)
```

---

## Computation Graphs

PyTorch uses **dynamic computation graphs** (define-by-run), building the graph as operations execute.

### Dynamic vs Static Graphs

| Aspect | Dynamic (PyTorch) | Static (TensorFlow 1.x) |
|--------|-------------------|-------------------------|
| **When built** | During execution | Before execution |
| **Flexibility** | Can change each iteration | Fixed structure |
| **Debugging** | Easy (Python debugger works) | Harder (graph is opaque) |
| **Performance** | Good (optimized on-the-fly) | Better (full optimization) |

### Example: Building a Graph

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward pass builds graph
z = x * y       # z depends on x and y
w = z + x**2    # w depends on z (and transitively on x, y)

print(z.grad_fn)  # <MulBackward0>
print(w.grad_fn)  # <AddBackward0>
```

**Graph structure**:
```
     x (leaf)        y (leaf)
      ↓               ↓
      └───── * ───────┘
             ↓
             z
             ↓              x (reused)
             └──── + ────────┘ (x²)
                    ↓
                    w
```

### Leaf Tensors vs Intermediate Tensors

**Leaf tensor**: Created by the user, not by operations
```python
x = torch.tensor(1.0, requires_grad=True)  # Leaf
y = torch.randn(3, requires_grad=True)     # Leaf
```

**Intermediate tensor**: Result of operations
```python
z = x + y  # Intermediate (has grad_fn)
```

**Key difference**:
- **Leaf tensors** store gradients in `.grad`
- **Intermediate tensors** don't store gradients (only leaf tensors need them for optimization)

---

## The Backward Pass

### Basic Backward

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3  # y = x³

y.backward()  # Compute dy/dx

print(x.grad)  # tensor(12.) = 3 * x² = 3 * 2² = 12
```

**What happened**:
1. Forward: `y = x³` computed
2. Graph: Recorded that `y` depends on `x` via power operation
3. Backward: Computed `dy/dx = 3x²`, stored in `x.grad`

### Multi-Output Backward

For vector outputs, you need to specify which output to backpropagate:

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2  # Element-wise: y = [x₀², x₁², x₂²]

# Need to specify gradient vector (which output to backprop)
gradient = torch.tensor([1.0, 1.0, 1.0])
y.backward(gradient)

print(x.grad)  # [2*x₀, 2*x₁, 2*x₂]
```

**Why?** Backprop needs a scalar loss. The `gradient` argument is the chain rule's "upstream gradient":

$$\frac{\partial \text{loss}}{\partial x_i} = \sum_j \frac{\partial \text{loss}}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$

The `gradient` is $\frac{\partial \text{loss}}{\partial y_j}$.

### Scalar Loss (Common Case)

In neural networks, the loss is scalar:

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2
loss = y.sum()  # Scalar

loss.backward()  # No argument needed!
print(x.grad)
```

---

## Gradient Flow and Accumulation

### Gradient Accumulation

By default, calling `.backward()` **accumulates** gradients:

```python
x = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"Iteration {i+1}: x.grad = {x.grad}")

# Output:
# Iteration 1: x.grad = 4.0
# Iteration 2: x.grad = 8.0   ← accumulated!
# Iteration 3: x.grad = 12.0
```

**Solution**: Zero gradients before each backward pass:

```python
x = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"Iteration {i+1}: x.grad = {x.grad}")
    x.grad.zero_()  # Zero out gradient
```

**Why accumulate by default?**
- Useful for gradient accumulation across mini-batches
- Needed for multi-task learning scenarios

### Detaching from the Graph

Sometimes you want to stop gradients from flowing through part of the graph:

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2

# Detach y from the graph
z = y.detach() + x

# Backward through z won't compute gradients w.r.t. y
loss = z.sum()
loss.backward()

print(x.grad)  # Only gradient from x term, not y term
```

**Use cases**:
- Freezing parts of a model
- Implementing stop-gradient in certain algorithms
- Computing metrics without affecting gradients

---

## Context Managers

PyTorch provides context managers to control gradient computation:

### `torch.no_grad()`

Disables gradient tracking (reduces memory, faster):

```python
x = torch.randn(3, requires_grad=True)

with torch.no_grad():
    y = x ** 2  # No gradient tracking
    print(y.requires_grad)  # False

# Outside context, tracking resumes
z = x ** 3
print(z.requires_grad)  # True
```

**Common use**: Evaluation/inference

```python
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

### `torch.inference_mode()`

Stronger than `no_grad()` – disables autograd entirely:

```python
with torch.inference_mode():
    y = x ** 2  # Even faster than no_grad()
```

**Difference from `no_grad()`**:
- `inference_mode()`: Cannot re-enable gradients inside the context
- `no_grad()`: Can nest with `enable_grad()` to re-enable

**Use `inference_mode()`** when you're sure you won't need gradients at all (most inference cases).

### `torch.enable_grad()`

Re-enables gradients (rarely needed):

```python
with torch.no_grad():
    # Gradients disabled

    with torch.enable_grad():
        # Gradients re-enabled
        y = x ** 2
        y.backward()
```

### `torch.set_grad_enabled()`

Conditional gradient tracking:

```python
is_training = True

with torch.set_grad_enabled(is_training):
    y = model(x)

# Equivalent to:
if is_training:
    y = model(x)
else:
    with torch.no_grad():
        y = model(x)
```

---

## Custom Autograd Functions

For operations not in PyTorch or needing custom gradients, use `torch.autograd.Function`:

### Template

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx: context object to save values for backward
        ctx.save_for_backward(input)

        # Compute forward pass
        output = # ... your computation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: gradient of loss w.r.t. output
        input, = ctx.saved_tensors

        # Compute gradient of loss w.r.t. input
        grad_input = # ... your gradient computation
        return grad_input
```

### Example: Custom ReLU

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0  # Gradient is 0 where input < 0
        return grad_input

# Usage
my_relu = MyReLU.apply

x = torch.randn(5, requires_grad=True)
y = my_relu(x)
y.sum().backward()
print(x.grad)
```

### Example: Straight-Through Estimator

Used for non-differentiable operations (e.g., quantization):

```python
class StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward: round to nearest integer
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: pass gradient through unchanged
        return grad_output

quantize = StraightThrough.apply

x = torch.tensor([1.2, 2.7], requires_grad=True)
y = quantize(x)  # tensor([1., 3.])
y.sum().backward()
print(x.grad)    # tensor([1., 1.]) - gradient passed through
```

### Multiple Inputs/Outputs

```python
class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        # ∂(a*b)/∂a = b
        # ∂(a*b)/∂b = a
        grad_a = grad_output * b
        grad_b = grad_output * a

        return grad_a, grad_b

multiply = Multiply.apply

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
z = multiply(x, y)
z.backward()

print(x.grad)  # tensor(4.) = y
print(y.grad)  # tensor(3.) = x
```

---

## Debugging Gradients

### Problem 1: NaN or Inf Gradients

**Symptom**: Model diverges, loss becomes NaN

**Diagnosis**:
```python
# Check for NaN/Inf in gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf gradient in {name}")
```

**Common causes**:
- Exploding gradients (use gradient clipping)
- Division by zero
- Log of zero/negative number
- Numerical instability (use stable implementations)

**Solution**: Enable anomaly detection

```python
with torch.autograd.detect_anomaly():
    loss.backward()
# Will raise error at the operation that produced NaN
```

### Problem 2: Vanishing Gradients

**Symptom**: Gradients become very small, early layers don't learn

**Diagnosis**:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: mean={param.grad.abs().mean():.6f}, max={param.grad.abs().max():.6f}")
```

**Solutions**:
- Better initialization (Xavier, He)
- Residual connections (ResNet)
- Batch normalization
- Different activation (ReLU instead of sigmoid)

### Problem 3: No Gradient

**Symptom**: `param.grad` is `None`

**Causes**:
```python
# 1. Parameter not in computation graph
x = torch.randn(10, requires_grad=True)
y = torch.randn(10, requires_grad=False)  # No gradient!
loss = (x + y).sum()
loss.backward()  # y.grad will be None

# 2. Detached from graph
x = torch.randn(10, requires_grad=True)
y = x.detach()  # Breaks connection
loss = y.sum()
loss.backward()  # x.grad will be None

# 3. Inside no_grad() context
with torch.no_grad():
    loss = model(x)
    loss.backward()  # No gradients computed
```

### Gradient Checking (Numerical Gradient)

Verify custom autograd functions:

```python
from torch.autograd import gradcheck

def test_custom_function():
    input = torch.randn(5, dtype=torch.double, requires_grad=True)
    test = gradcheck(MyFunction.apply, input, eps=1e-6)
    print(f"Gradient check passed: {test}")
```

### Gradient Hooks

Inspect or modify gradients during backprop:

```python
def print_grad(grad):
    print(f"Gradient: {grad}")
    return grad  # Can modify gradient here

x = torch.randn(3, requires_grad=True)
x.register_hook(print_grad)

y = x ** 2
y.sum().backward()  # Will print gradient
```

**Use cases**:
- Debugging gradient flow
- Gradient clipping per-tensor
- Monitoring gradient statistics

---

## Advanced Topics

### Higher-Order Gradients

Compute gradients of gradients:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

# First-order gradient
grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
print(grad_y)  # 12.0 = 3*x²

# Second-order gradient
grad2_y = torch.autograd.grad(grad_y, x)[0]
print(grad2_y)  # 12.0 = 6*x
```

**Use case**: Hessian computation, meta-learning

### Jacobian and Hessian

```python
from torch.autograd.functional import jacobian, hessian

def f(x):
    return x ** 3

x = torch.tensor([1.0, 2.0, 3.0])

jac = jacobian(f, x)
print(jac)  # Diagonal: [3, 12, 27] = [3*1², 3*2², 3*3²]

hess = hessian(f, x)
print(hess.diag())  # [6, 12, 18] = [6*1, 6*2, 6*3]
```

### Gradient Checkpointing (Memory Optimization)

Trade compute for memory by recomputing activations during backward:

```python
from torch.utils.checkpoint import checkpoint

def expensive_function(x):
    # Many operations...
    return x ** 2

x = torch.randn(1000, 1000, requires_grad=True)

# Regular: stores all intermediate activations
y = expensive_function(x)

# Checkpointed: only stores input, recomputes during backward
y = checkpoint(expensive_function, x)
```

**Use case**: Training very deep networks or large inputs

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Dynamic Graph** | Built during forward pass, specific to each input |
| **`.backward()`** | Compute gradients via reverse-mode autodiff |
| **Leaf Tensors** | User-created tensors; gradients stored in `.grad` |
| **`requires_grad`** | Enable/disable gradient tracking |
| **`no_grad()`** | Disable tracking (inference, evaluation) |
| **Custom Functions** | `torch.autograd.Function` for custom ops |

### Mental Model

```
Forward:    Operations build a computation graph
            ↓
Backward:   .backward() traverses graph in reverse,
            computing gradients via chain rule
            ↓
Result:     Gradients stored in leaf tensor .grad
```

### Common Patterns

```python
# Training loop pattern
optimizer.zero_grad()        # 1. Clear old gradients
output = model(input)        # 2. Forward pass
loss = criterion(output)     # 3. Compute loss
loss.backward()              # 4. Backward pass
optimizer.step()             # 5. Update parameters

# Evaluation pattern
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

### Troubleshooting Checklist

- ✓ Did you call `optimizer.zero_grad()`?
- ✓ Is `requires_grad=True` on inputs/parameters?
- ✓ Did you accidentally use `.detach()` or `no_grad()`?
- ✓ Is the computation graph broken (in-place ops)?
- ✓ Are gradients NaN/Inf (numerical issues)?
- ✓ Are gradients vanishing (check magnitudes)?

---

**Previous**: [01_tensors.md](01_tensors.md) - Tensor fundamentals
**Next**: [03_modules.md](03_modules.md) - Building neural networks

**Related**:
- [Quick Reference](quick_reference.md) - Autograd API cheat sheet
- [Appendix: Debugging Guide](appendix_debugging_guide.md) - Systematic debugging
