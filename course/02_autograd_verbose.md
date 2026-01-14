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

**Automatic differentiation**, commonly called **autograd**, is PyTorch's system for computing derivatives of tensor operations automatically. It forms the essential computational engine that powers neural network training through backpropagation, allowing you to train models without ever manually deriving or implementing gradient formulas. In essence, autograd is what transforms PyTorch from a simple numerical computing library into a powerful deep learning framework.

The fundamental challenge that autograd solves is this: neural networks are mathematical functions composed of millions of operations (matrix multiplies, activations, normalizations, etc.), and training them requires computing the derivative of a loss function with respect to every single parameter in the network. Doing this by hand would be impossibly tedious and error-prone—you'd need to derive gradient formulas using calculus, implement those formulas in code, and then debug the inevitable mistakes. For complex architectures with branching paths, residual connections, and dynamic behavior, manual gradient computation would be practically infeasible.

### Why Autograd Matters: The Transformation It Enables

Without automatic differentiation, training a neural network would require you to:

1. **Manually derive gradient formulas** for every operation in your network using calculus and the chain rule. For a simple fully connected layer, this is tedious but doable. For a ResNet with skip connections or a Transformer with attention mechanisms, this becomes nightmarishly complex.

2. **Implement backward passes by hand** for every layer and operation. You'd write functions that compute how much each input contributes to the loss, being extremely careful about matrix dimensions, broadcasting rules, and numerical stability.

3. **Debug complex derivative chains** when something goes wrong. When your network doesn't converge, is it a bug in your forward pass, your backward pass, your gradient formula derivation, or just poor hyperparameters? The possibilities multiply, making development painfully slow.

With autograd, PyTorch eliminates all three of these burdens through a remarkably elegant system:

- **Records operations automatically** as you compute in the forward pass, building a data structure that remembers what happened
- **Builds a computation graph** that represents the dependencies between tensors and operations
- **Computes gradients automatically** when you call `.backward()`, traversing the graph in reverse and applying the chain rule at each step

This automation is not just a convenience—it's what makes modern deep learning research possible. Researchers can experiment with novel architectures and loss functions without worrying about deriving gradients, which accelerates the pace of innovation dramatically.

### The Big Picture: Forward and Backward Passes

The fundamental workflow of autograd can be visualized as two complementary passes through your computation:

```
Forward pass:   Input → Operations → Output
                (PyTorch records each operation in a graph)

Backward pass:  Output ← Gradients ← Input
                (.backward() traverses the graph computing gradients)
```

During the **forward pass**, you simply write normal Python code that performs computations on tensors—adding them, multiplying them, applying functions like ReLU or softmax. As these operations execute, PyTorch silently records them in a computation graph structure. This graph captures the dependencies: which tensors depend on which other tensors, and through which operations.

During the **backward pass**, which you trigger by calling `.backward()` on your loss tensor, PyTorch traverses this graph in reverse topological order (from outputs back to inputs) and applies the chain rule of calculus at each node. The chain rule tells us how to propagate gradients backward: if `z = f(y)` and `y = g(x)`, then `dz/dx = (dz/dy) * (dy/dx)`. PyTorch automates this chaining process, accumulating the total gradient with respect to each parameter as it goes.

The beauty of this design is that you write your forward pass in intuitive, imperative Python code that directly expresses what you want to compute, and backpropagation happens automatically without you having to think about it.

---

## Computation Graphs

At the heart of autograd is the **computation graph**—a directed acyclic graph (DAG) where nodes represent tensors and edges represent operations that produced those tensors. Understanding how PyTorch builds and uses this graph is essential for understanding how gradients flow through your network and for debugging gradient-related issues.

### Dynamic vs Static Graphs: PyTorch's Key Design Choice

PyTorch uses **dynamic computation graphs**, also called **define-by-run** graphs, which is a fundamental architectural decision that distinguishes it from some other frameworks. Understanding this choice and its implications helps you appreciate PyTorch's design philosophy and understand when to use different frameworks.

| Aspect | Dynamic (PyTorch, JAX eager) | Static (TensorFlow 1.x, older frameworks) |
|--------|------------------------------|-------------------------------------------|
| **When graph is built** | During execution (at runtime) | Before execution (compilation phase) |
| **Graph structure** | Can change every iteration | Fixed structure determined upfront |
| **Flexibility** | Extremely flexible (control flow, different architectures per batch) | Limited (must know structure in advance) |
| **Debugging** | Easy (Python debugger works, stack traces are meaningful) | Harder (graph is opaque, errors occur at execution not definition) |
| **Performance** | Good (optimizations happen JIT) | Potentially better (full-program optimization possible) |
| **Pythonic** | Very (just write Python code) | Less (need to learn graph construction API) |

The **dynamic graph** approach means that every time you run a forward pass, PyTorch builds a fresh computation graph that reflects exactly what happened during that particular execution. This has profound implications:

- **Control flow just works**: You can use normal Python if-statements, for-loops, and while-loops in your model, and the graph will reflect the path actually taken. Different inputs can trigger different computational paths through your model without any special handling.

- **Debugging is natural**: When something goes wrong, you get a normal Python stack trace pointing to the exact line of code that caused the problem. You can use `print()` statements, breakpoints, and Python debuggers directly in your model code.

- **Dynamic architectures are easy**: Models that change structure based on input (like recursive neural networks that process trees, or models with input-dependent depth) are straightforward to implement because the graph is built dynamically.

The tradeoff is that dynamic graphs have slightly more overhead (building the graph each time has a cost) and miss some optimization opportunities (the framework can't do whole-program optimization if it doesn't know the graph structure in advance). However, for most use cases, the flexibility and ease of development far outweigh these costs, which is why dynamic graphs have become the dominant paradigm in modern deep learning frameworks.

### Example: Building a Computation Graph

Let's trace through how PyTorch builds a graph as you perform operations:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward pass builds the graph operation by operation
z = x * y       # z depends on x and y through multiplication
w = z + x**2    # w depends on z (and transitively on x, y) and on x squared

print(z.grad_fn)  # <MulBackward0> - remembers multiplication created z
print(w.grad_fn)  # <AddBackward0> - remembers addition created w
```

**Graph structure** (visualized):
```
     x (leaf)        y (leaf)
      ↓               ↓
      └───── * ───────┘
             ↓
             z (MulBackward0)
             ↓              x (reused)
             └──── + ────────┘ (x²  via PowBackward0)
                    ↓
                    w (AddBackward0)
```

Notice several important aspects of this graph:

1. **`grad_fn` attribute**: Every tensor that results from an operation has a `grad_fn` attribute that points to the function object representing the operation that created it. This is how PyTorch remembers the operation for later gradient computation. The names like `MulBackward0` and `AddBackward0` are internal PyTorch classes that know how to compute gradients for those operations.

2. **Leaf tensors have no `grad_fn`**: The tensors `x` and `y` were created directly by you (via `torch.tensor()`), not by operations on other tensors. These are called **leaf tensors**, and they're special because they're the starting points of the computation. Leaf tensors don't have a `grad_fn` because no operation created them.

3. **Variables can be reused**: Notice that `x` appears twice in the computation of `w`—once through `z` (which is `x * y`) and once directly (as `x**2`). This is perfectly fine and common in neural networks (think of skip connections in ResNets). When backpropagation happens, gradients will flow through both paths and be summed at `x`.

4. **The graph captures dependencies**: The structure of the graph encodes which tensors depend on which others. When we eventually call `.backward()` on `w`, PyTorch will know it needs to compute gradients with respect to both `x` and `y`, and it will know how those gradients flow through the multiplication and addition operations.

### Leaf Tensors vs Intermediate Tensors: A Crucial Distinction

Understanding the difference between **leaf tensors** and **intermediate tensors** is essential for understanding where gradients are stored and why.

**Leaf tensors** are tensors created directly by you, not as the result of operations on other tensors:

```python
x = torch.tensor(1.0, requires_grad=True)  # Leaf tensor
y = torch.randn(3, requires_grad=True)     # Leaf tensor
param = torch.nn.Parameter(torch.randn(10))  # Also a leaf (parameters are leaves)
```

**Intermediate tensors** are tensors that result from operations on other tensors:

```python
z = x + y  # Intermediate tensor - result of addition
w = z.relu()  # Intermediate tensor - result of ReLU
```

The **key differences** that matter in practice:

1. **Gradient storage**: Leaf tensors store their gradients in the `.grad` attribute after you call `.backward()`. Intermediate tensors do not store their gradients by default—the gradient with respect to an intermediate tensor is computed during backpropagation but then discarded after being used to compute gradients for earlier tensors.

2. **Why this design?**: In optimization, you only need gradients with respect to the **parameters** of your model (which are leaf tensors), not with respect to every intermediate activation. Storing gradients for every intermediate tensor would waste enormous amounts of memory. For example, in a ResNet-50 processing a batch of images, there are hundreds of intermediate tensors, but you only need to update the ~25 million parameter tensors.

3. **Retaining gradients for intermediates**: If you do need the gradient with respect to an intermediate tensor (for debugging or certain advanced algorithms), you can call `.retain_grad()` on it:

```python
z = x + y
z.retain_grad()  # Tell PyTorch to save z's gradient
loss = (z ** 2).sum()
loss.backward()
print(z.grad)  # Now this works!
```

This distinction might seem like an implementation detail, but it's actually a performance optimization that makes training large models feasible. Without it, PyTorch would need to store gradients for every single tensor in your computation, which could easily exceed your GPU memory.

---

## The Backward Pass

The backward pass is where the magic happens—where PyTorch computes all the gradients automatically using the computation graph it built during the forward pass. Understanding how `.backward()` works and its variants is essential for effective PyTorch usage.

### Basic Backward: The Simplest Case

Let's start with the simplest possible case: a scalar output that depends on a single input:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3  # y = x³

y.backward()  # Compute dy/dx

print(x.grad)  # tensor(12.) = 3 * x² = 3 * 2² = 12
```

**What happened step by step:**

1. **Forward pass**: PyTorch computed `y = x³` and stored the result (8.0). Simultaneously, it recorded in the computation graph that `y` was created by raising `x` to the power 3, storing a reference to the `PowBackward` operation.

2. **Graph structure**: The graph now contains the dependency: `x (leaf) → PowBackward(exponent=3) → y`.

3. **Backward pass**: When you called `y.backward()`, PyTorch:
   - Started at `y` (the output)
   - Applied the derivative rule for power functions: d(x³)/dx = 3x²
   - Evaluated this at x=2: 3 * 2² = 12
   - Stored the result in `x.grad`

The crucial insight is that you didn't have to know or implement the derivative formula for the power operation—PyTorch knows the derivatives for all its built-in operations and applies them automatically.

### The Chain Rule: Composition of Functions

Most neural networks involve many layers of operations composed together. The chain rule is what allows us to compute gradients through these compositions:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2      # y = x²
z = y + 3       # z = y + 3 = x² + 3
w = z ** 2      # w = z² = (x² + 3)²

w.backward()
print(x.grad)   # How much does w change when x changes?
```

The gradient `dw/dx` requires chaining together the derivatives:
- `dw/dz = 2z = 2(x² + 3)`
- `dz/dy = 1`
- `dy/dx = 2x`
- Therefore: `dw/dx = (dw/dz) * (dz/dy) * (dy/dx) = 2(x² + 3) * 1 * 2x`

PyTorch performs this chaining automatically, traversing the graph backward from `w` to `x` and multiplying the local gradients at each step.

### Multi-Output Backward: Handling Vector Outputs

The simple `.backward()` call works when your output is a scalar (a single number). But what if your output is a vector or a tensor with multiple elements? This is where things get slightly more complex, and understanding it is crucial for advanced use cases.

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2  # Element-wise: y = [x₀², x₁², x₂²]

# This would error: y.backward()  ← can't backward through non-scalar
# Instead, need to specify a gradient tensor:
gradient = torch.tensor([1.0, 1.0, 1.0])
y.backward(gradient)

print(x.grad)  # [2*x₀, 2*x₁, 2*x₂]
```

**Why is the `gradient` argument needed?** The mathematical reason is that `.backward()` computes the gradient of a scalar with respect to the inputs. When you have a vector output `y = [y₀, y₁, y₂]`, there's no single scalar to take the gradient of—there are three separate scalars. The `gradient` argument tells PyTorch which weighted combination of outputs you care about.

Mathematically, what's actually being computed is:

$$\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$

where the `gradient` tensor is $\frac{\partial L}{\partial y_j}$ (the "upstream gradient" from whatever loss function would eventually consume `y`). When you pass `gradient = torch.tensor([1.0, 1.0, 1.0])`, you're saying "treat all outputs equally," which is equivalent to taking the gradient of `y.sum()`.

### The Common Case: Scalar Loss Functions

In practice, the multi-output case rarely comes up in neural network training because your loss function reduces everything to a single scalar:

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2         # Vector
loss = y.sum()     # Scalar loss - sum of squared values

loss.backward()    # No argument needed!
print(x.grad)      # [2*x₀, 2*x₁, 2*x₂]
```

This is the standard pattern: compute some vector or tensor of outputs (like predictions for a batch of examples), compute a scalar loss function that measures how good those predictions are, and call `.backward()` on that scalar loss. The gradient argument is implicitly 1.0 for the scalar case.

### Multiple Backward Calls: A Common Pitfall

A subtle but important point: calling `.backward()` consumes the computation graph by default:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

y.backward()  # First backward - works fine
# y.backward()  # Second backward - ERROR: graph has been freed!
```

This is a memory optimization—once you've computed gradients, PyTorch frees the graph to save memory. If you need to backward through the same graph multiple times (rare in standard training, but useful in some meta-learning scenarios), you must pass `retain_graph=True`:

```python
y.backward(retain_graph=True)  # Keep graph
y.backward()  # Now this works
```

---

## Gradient Flow and Accumulation

Understanding how gradients accumulate and flow through your computation graph is essential for correct training loops and for diagnosing gradient-related bugs.

### Gradient Accumulation: The Default Behavior

One of PyTorch's design choices that surprises newcomers is that calling `.backward()` **accumulates gradients** rather than overwriting them:

```python
x = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"Iteration {i+1}: x.grad = {x.grad}")

# Output:
# Iteration 1: x.grad = 4.0
# Iteration 2: x.grad = 8.0   ← 4.0 + 4.0 (accumulated!)
# Iteration 3: x.grad = 12.0  ← 8.0 + 4.0 (accumulated again!)
```

**Why is this the default?** This behavior might seem like a bug, but it's actually a deliberate design choice that supports important use cases:

1. **Gradient accumulation across mini-batches**: When your full batch doesn't fit in memory, you can split it into smaller micro-batches, accumulate gradients across them, and then update parameters once. This is mathematically equivalent to training on the full batch but uses less memory.

2. **Multi-task learning**: If you have multiple loss functions (like in multi-task learning or GANs), you can backward through each one and accumulate the total gradient before updating.

3. **Efficient computation patterns**: Some advanced training techniques intentionally accumulate gradients across multiple forward passes.

**The solution**: In a standard training loop, you must **zero the gradients** before each backward pass:

```python
x = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    x.grad = None  # Option 1: set to None (slightly more efficient)
    # OR
    x.grad.zero_()  # Option 2: zero in-place (if grad exists)

    y = x ** 2
    y.backward()
    print(f"Iteration {i+1}: x.grad = {x.grad}")
```

In practice, you'll almost always use the optimizer's `.zero_grad()` method which handles this for all parameters:

```python
optimizer.zero_grad()  # Clear gradients for all parameters
output = model(input)
loss = criterion(output, target)
loss.backward()        # Compute gradients
optimizer.step()       # Update parameters
```

### Detaching Tensors: Stopping Gradient Flow

Sometimes you want to use a tensor's value in a computation but **stop gradients from flowing through it**. This is what `.detach()` does—it creates a new tensor that shares the same data but is disconnected from the computation graph:

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2

# Detach y from the graph
z = y.detach() + x

# Backward through z computes gradients w.r.t. x (from the "x" term)
# but NOT through y (because y was detached)
loss = z.sum()
loss.backward()

print(x.grad)  # Only gradient from the "x" term in z = y.detach() + x
```

**Important use cases for detaching:**

1. **Freezing parts of a model**: When you want to train one part of a model while keeping another part fixed, you detach the outputs of the frozen part to prevent gradients from flowing backward into it.

2. **Stop-gradient operations**: Some algorithms (like target networks in reinforcement learning or momentum encoders in contrastive learning) specifically require stopping gradient flow at certain points in the computation.

3. **Computing metrics**: When you want to compute statistics or metrics without those computations affecting gradients:

```python
with torch.no_grad():
    accuracy = (predictions.argmax(1) == labels).float().mean()
    # accuracy is detached, doesn't affect gradients
```

4. **Implementing custom gradient logic**: In advanced scenarios, you might compute something in the forward pass but want to implement a custom gradient for it rather than using autograd's automatic gradient.

**Detaching vs `.data` (legacy)**: In older PyTorch code, you might see `.data` used for a similar purpose. However, `.data` is now discouraged because it can lead to subtle bugs—use `.detach()` instead, which is safer and more explicit.

---

## Context Managers

PyTorch provides several context managers that control whether and how gradients are computed. Understanding these is essential for writing efficient code and for implementing certain algorithms correctly.

### `torch.no_grad()`: Disabling Gradient Tracking

The **`torch.no_grad()`** context manager temporarily disables gradient tracking for all operations inside the context:

```python
x = torch.randn(3, requires_grad=True)

with torch.no_grad():
    y = x ** 2  # This operation is not tracked
    print(y.requires_grad)  # False - no gradient tracking

# Outside the context, tracking resumes
z = x ** 3
print(z.requires_grad)  # True - tracking is back on
```

**Why would you disable gradients?** Two main reasons:

1. **Memory efficiency**: Building and storing the computation graph consumes memory. For intermediate tensors in a deep network, this memory overhead can be substantial. During inference (evaluation), you don't need gradients at all, so disabling tracking saves memory—often allowing you to use much larger batch sizes during evaluation than during training.

2. **Computational efficiency**: When gradients aren't being tracked, PyTorch can skip bookkeeping operations like storing intermediate activations for backward passes and building the computation graph. This makes operations somewhat faster (typically 10-30% for inference workloads).

**The standard evaluation pattern:**

```python
model.eval()  # Set model to evaluation mode (affects dropout, batch norm)
with torch.no_grad():
    for data in test_loader:
        predictions = model(data)
        # Compute metrics, etc.
```

This pattern has become so standard that many practitioners write it automatically without thinking about it. The `model.eval()` call changes the behavior of certain layers (disabling dropout, freezing batch normalization statistics), while `torch.no_grad()` disables gradient computation entirely.

### `torch.inference_mode()`: Even Stronger Optimization

Introduced in PyTorch 1.9, **`torch.inference_mode()`** is a more aggressive version of `no_grad()` that disables autograd entirely (not just gradient tracking):

```python
with torch.inference_mode():
    y = x ** 2  # Even faster than no_grad()
```

**The difference from `no_grad()`:**

- **`torch.no_grad()`**: Disables gradient tracking but still allows you to nest it with `torch.enable_grad()` to selectively re-enable tracking. This flexibility has a tiny performance cost.

- **`torch.inference_mode()`**: Completely disables the autograd engine, which allows for more aggressive optimizations. However, you cannot re-enable gradients inside this context.

**When to use which:**

```python
# Use inference_mode() for pure inference (most common)
model.eval()
with torch.inference_mode():
    predictions = model(test_data)

# Use no_grad() when you might need to selectively re-enable gradients
with torch.no_grad():
    frozen_features = frozen_model(input)
    with torch.enable_grad():
        # This specific computation needs gradients
        loss = trainable_head(frozen_features)
        loss.backward()
```

In practice, for straightforward inference, `inference_mode()` is the better choice because it's faster. Use `no_grad()` only when you need the flexibility to nest gradient contexts.

### `torch.enable_grad()`: Selective Re-enabling

The **`torch.enable_grad()`** context manager re-enables gradient tracking inside a `no_grad()` context:

```python
x = torch.randn(3, requires_grad=True)

with torch.no_grad():
    # Gradients disabled here
    y = x ** 2

    with torch.enable_grad():
        # Gradients re-enabled for this block
        z = x ** 3
        z.sum().backward()  # This works!
        print(x.grad)  # Gradients computed
```

This is rarely needed in standard training loops but is useful in advanced scenarios like:
- Training meta-learning models where you need fine-grained control over which computations are differentiated
- Implementing algorithms that mix gradient-based and gradient-free components
- Debugging specific parts of a large model by selectively enabling gradients

### `torch.set_grad_enabled()`: Conditional Gradient Tracking

The **`torch.set_grad_enabled()`** context manager allows you to conditionally enable or disable gradients based on a boolean:

```python
is_training = True  # Could be a flag from command-line args, etc.

with torch.set_grad_enabled(is_training):
    output = model(input)
    # Gradients enabled if is_training=True, disabled otherwise

# This is cleaner than writing:
if is_training:
    output = model(input)
else:
    with torch.no_grad():
        output = model(input)
```

This pattern is particularly useful when writing code that needs to work in both training and evaluation modes without duplicating logic. Many training frameworks use this internally to implement the train/eval distinction.

---

## Custom Autograd Functions

While PyTorch provides gradients for hundreds of operations out of the box, sometimes you need to implement a custom operation with its own gradient logic. This might be because you're implementing a novel operation, integrating with external libraries, or need to customize how gradients flow for algorithmic reasons.

### The `torch.autograd.Function` Template

Custom autograd functions inherit from **`torch.autograd.Function`** and implement two static methods: `forward()` and `backward()`:

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx: context object to save values needed for backward
        ctx.save_for_backward(input)

        # Compute and return the forward pass result
        output = # ... your computation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: gradient of loss w.r.t. this function's output
        #              (the "upstream gradient" from chain rule)

        # Retrieve saved tensors
        input, = ctx.saved_tensors

        # Compute gradient of loss w.r.t. input
        grad_input = # ... your gradient computation using chain rule

        # Return gradient w.r.t. each input argument (None for non-tensor args)
        return grad_input
```

**Key points about this pattern:**

1. **`ctx` (context object)**: This is a special object that serves as communication channel between `forward()` and `backward()`. You use `ctx.save_for_backward()` to store tensors that you'll need when computing gradients. You can also store non-tensor values with `ctx.constant = value`.

2. **`grad_output` in backward**: This is the gradient of the loss with respect to your function's output. In the chain rule, this is the "upstream gradient" that you need to multiply with your local gradients. If your function returns multiple outputs, you'll receive multiple `grad_output` tensors.

3. **Return values**: The `backward()` method must return one gradient tensor per input tensor from `forward()`. If an input doesn't require gradients (like integer indices), return `None` for that input.

### Example: Custom ReLU Implementation

Let's implement ReLU (Rectified Linear Unit) to see how this works in practice:

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save input for backward (we need to know which elements were positive)
        ctx.save_for_backward(input)

        # ReLU forward: max(0, x) for each element
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        input, = ctx.saved_tensors

        # Create gradient tensor (start with a copy of upstream gradient)
        grad_input = grad_output.clone()

        # ReLU gradient: 1 if input > 0, else 0
        # (zero out gradients where input was negative)
        grad_input[input < 0] = 0

        return grad_input

# Usage: call via .apply()
my_relu = MyReLU.apply

x = torch.randn(5, requires_grad=True)
y = my_relu(x)
y.sum().backward()
print(x.grad)  # Non-zero only where x was positive
```

**Understanding the gradient:** ReLU's derivative is simple:
- If input > 0, gradient passes through unchanged (derivative is 1)
- If input ≤ 0, gradient is blocked (derivative is 0)

This is implemented by masking `grad_output`: we keep gradients where input was positive and zero them where input was negative.

### Example: Straight-Through Estimator

A more interesting example is the **straight-through estimator**, used for non-differentiable operations like quantization:

```python
class StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward: round to nearest integer (non-differentiable!)
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: pretend the operation was identity
        # (pass gradient through unchanged)
        return grad_output

quantize = StraightThrough.apply

x = torch.tensor([1.2, 2.7, -0.3], requires_grad=True)
y = quantize(x)  # tensor([1., 3., -0.])
y.sum().backward()
print(x.grad)    # tensor([1., 1., 1.]) - gradient passed through!
```

This is a "lie" to the autograd system—the forward pass has zero derivative almost everywhere (it's piecewise constant), but we tell backpropagation that the derivative is 1. This is useful for training networks with quantized activations or discrete decisions, where the true gradient would be unhelpful (zero almost everywhere) but a surrogate gradient allows learning.

### Example: Multiple Inputs and Outputs

Custom functions can handle multiple inputs and outputs:

```python
class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # Save both inputs for backward
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        a, b = ctx.saved_tensors

        # Apply chain rule:
        # ∂(a*b)/∂a = b
        # ∂(a*b)/∂b = a
        grad_a = grad_output * b
        grad_b = grad_output * a

        # Return one gradient per input
        return grad_a, grad_b

multiply = Multiply.apply

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
z = multiply(x, y)  # z = 12
z.backward()

print(x.grad)  # tensor(4.) = y
print(y.grad)  # tensor(3.) = x
```

### When to Use Custom Autograd Functions

Custom autograd functions are needed in several scenarios:

1. **Implementing novel operations**: When you're researching new architectural components not yet in PyTorch
2. **Interfacing with external code**: When you need to call C++/CUDA libraries and define how gradients flow through them
3. **Customizing gradient behavior**: When you want different gradient behavior than the natural mathematical derivative (like straight-through estimators)
4. **Numerical stability**: When the automatic gradient would be numerically unstable, you can implement a more stable version
5. **Performance optimization**: When you know a more efficient way to compute gradients than the automatic version

---

## Debugging Gradients

Gradient-related bugs are among the most common and frustrating issues in deep learning. Understanding how to systematically diagnose and fix them is essential for productive development.

### Problem 1: NaN or Inf Gradients (Numerical Instability)

**Symptom**: Your loss suddenly becomes `NaN`, or training diverges with the loss shooting to infinity.

**Diagnosis**: Check for NaN or Inf values in gradients:

```python
# After loss.backward(), check all parameter gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"⚠️ NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"⚠️ Inf gradient in {name}")
```

**Common causes:**

1. **Exploding gradients**: In deep networks or RNNs, gradients can grow exponentially during backpropagation. Solution: Use **gradient clipping** to cap maximum gradient magnitude:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

2. **Division by zero**: Check for operations that might divide by zero, especially in normalization layers. Add small epsilon values: `x / (std + 1e-8)`.

3. **Log of zero or negative**: `log(0)` is negative infinity, `log(negative)` is NaN. Clamp inputs: `torch.log(x.clamp(min=1e-8))`.

4. **Numerical instability**: Some operations are mathematically correct but numerically unstable in floating-point. Use stable implementations (like `log_softmax` instead of `log(softmax)`).

**PyTorch's anomaly detection**: Enable this to pinpoint exactly which operation produced NaN:

```python
with torch.autograd.detect_anomaly():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()  # Will raise detailed error at the operation that produced NaN
```

This is slow, so only use it when debugging, not during normal training.

### Problem 2: Vanishing Gradients (Gradients Too Small)

**Symptom**: Early layers of your network don't learn; their parameters barely change, and the model only uses its later layers.

**Diagnosis**: Monitor gradient magnitudes across layers:

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_mean = param.grad.abs().mean().item()
        grad_max = param.grad.abs().max().item()
        print(f"{name}: mean={grad_mean:.6f}, max={grad_max:.6f}")
```

If early layers have gradients that are orders of magnitude smaller than later layers (like 1e-8 vs 1e-2), you have vanishing gradients.

**Common causes and solutions:**

1. **Deep networks with sigmoid/tanh activations**: These functions saturate (have near-zero derivative) for large inputs. Solution: Use **ReLU** or **GELU** activations instead.

2. **Poor weight initialization**: If weights are too small, activations and gradients shrink at each layer. Solution: Use **Xavier** or **He initialization** (PyTorch does this by default for most layers).

3. **No skip connections**: In very deep networks, gradients have to flow through many layers. Solution: Use **residual connections** (ResNet-style), which provide gradient highways.

4. **No normalization**: Activation magnitudes can shrink or explode. Solution: Use **Batch Normalization** or **Layer Normalization**.

### Problem 3: No Gradient (Tensor Has `None` Gradient)

**Symptom**: After calling `.backward()`, some parameter's `.grad` is `None` instead of a tensor.

**Common causes:**

```python
# Cause 1: Parameter not in computation graph
x = torch.randn(10, requires_grad=True)
y = torch.randn(10, requires_grad=False)  # No gradients!
loss = (x + y).sum()
loss.backward()
# x.grad exists, but y.grad is None (expected - requires_grad=False)

# Cause 2: Detached from graph
x = torch.randn(10, requires_grad=True)
y = x.detach()  # Breaks gradient flow
loss = y.sum()
loss.backward()
# x.grad will be None (gradient flow was cut by detach)

# Cause 3: Inside no_grad() context
x = torch.randn(10, requires_grad=True)
with torch.no_grad():
    loss = (x ** 2).sum()
loss.backward()  # Error or x.grad will be None
```

**Debugging strategy**: Trace backward from the loss to the parameter and check at each step:
- Is `requires_grad=True`?
- Is there a `.detach()` call?
- Are you inside `no_grad()`?
- Is the operation actually used in computing the loss?

### Gradient Checking: Numerical Verification

When implementing custom autograd functions, you should verify your gradient implementation against numerical gradients:

```python
from torch.autograd import gradcheck

def test_my_function():
    # Use double precision for numerical stability
    input = torch.randn(5, dtype=torch.double, requires_grad=True)

    # gradcheck compares your analytical gradient to numerical gradient
    test = gradcheck(MyFunction.apply, input, eps=1e-6, atol=1e-4)
    print(f"Gradient check passed: {test}")

test_my_function()
```

This computes numerical gradients using finite differences and compares them to your analytical gradients. If they don't match, your `backward()` implementation has a bug.

### Gradient Hooks: Inspecting Gradients During Backprop

For fine-grained gradient inspection or modification, use **hooks**:

```python
def inspect_grad(grad):
    print(f"Gradient: mean={grad.mean():.4f}, std={grad.std():.4f}")
    return grad  # Can modify gradient here if needed

x = torch.randn(100, requires_grad=True)
x.register_hook(inspect_grad)

y = (x ** 2).sum()
y.backward()  # Prints gradient statistics during backprop
```

**Use cases:**
- Debugging gradient flow by printing intermediate gradients
- Implementing per-tensor gradient clipping
- Monitoring gradient statistics for research insights
- Modifying gradients for certain algorithms (like applying masks)

---

## Advanced Topics

For advanced users and researchers, PyTorch's autograd system supports several sophisticated capabilities.

### Higher-Order Gradients: Gradients of Gradients

Sometimes you need to compute **second derivatives** (like the Hessian matrix) or even higher-order derivatives. PyTorch supports this through the `create_graph=True` option:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

# First derivative: dy/dx
grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"First derivative: {grad_y}")  # 12.0 = 3*x²

# Second derivative: d²y/dx²
grad2_y = torch.autograd.grad(grad_y, x)[0]
print(f"Second derivative: {grad2_y}")  # 12.0 = 6*x
```

**What's happening:** The `create_graph=True` flag tells PyTorch to build a computation graph for the gradient computation itself, allowing you to differentiate through the gradient.

**Use cases:**
- **Hessian computation**: For second-order optimization methods
- **Meta-learning**: Algorithms like MAML that differentiate through optimization steps
- **Physics-informed neural networks**: Which use PDEs involving second derivatives
- **Adversarial training**: Certain techniques require gradient penalties

### Jacobian and Hessian: Matrix Derivatives

For vector-valued functions, you might want the full Jacobian or Hessian matrices:

```python
from torch.autograd.functional import jacobian, hessian

def f(x):
    return x ** 3  # Element-wise cubing

x = torch.tensor([1.0, 2.0, 3.0])

# Jacobian: matrix of all partial derivatives
jac = jacobian(f, x)
print(jac)  # Diagonal matrix: [3, 12, 27] = [3*1², 3*2², 3*3²]

# Hessian: matrix of all second partial derivatives
hess = hessian(f, x)
print(hess.diag())  # Diagonal: [6, 12, 18] = [6*1, 6*2, 6*3]
```

For non-element-wise functions, these matrices capture how each output depends on each input. Computing full Jacobians and Hessians is expensive (O(n²) or worse), so these are typically only used for small models or specific research applications.

### Gradient Checkpointing: Trading Compute for Memory

Training very deep networks or processing very large inputs can exhaust GPU memory because PyTorch stores all intermediate activations for backpropagation. **Gradient checkpointing** is a technique that trades computation for memory:

```python
from torch.utils.checkpoint import checkpoint

def expensive_function(x):
    # Imagine this is a very deep network segment
    return x ** 2

x = torch.randn(1000, 1000, requires_grad=True)

# Normal: stores all intermediate activations (uses more memory)
y = expensive_function(x)

# Checkpointed: only stores input, recomputes activations during backward
y = checkpoint(expensive_function, x)
```

**How it works:**
- **Forward pass**: Runs the function normally but doesn't store intermediate activations
- **Backward pass**: Re-runs the forward pass to recompute activations, then immediately computes gradients

This approximately doubles training time (running forward pass twice) but can dramatically reduce memory usage, allowing you to train larger models or use bigger batches.

**When to use:**
- Very deep networks that don't fit in GPU memory
- Transformers with very long sequences
- Any situation where memory is the bottleneck and you can afford extra computation

---

## Summary

### Key Concepts Recap

| Concept | Core Understanding |
|---------|-------------------|
| **Automatic Differentiation** | PyTorch automatically computes gradients by recording operations in a computation graph during the forward pass, then traversing it backward applying the chain rule. |
| **Dynamic Computation Graph** | The graph is built at runtime (define-by-run), allowing arbitrary Python control flow and making debugging natural. New graph built for each forward pass. |
| **`.backward()`** | Triggers reverse-mode automatic differentiation, computing gradients via the chain rule. For scalar losses, call with no arguments; for vector outputs, pass a gradient tensor. |
| **Leaf Tensors** | Tensors created by the user (not by operations). Only leaves store gradients in `.grad`. Parameters are leaves. |
| **Intermediate Tensors** | Results of operations. Have `grad_fn` pointing to creating operation. Don't store gradients by default (memory optimization). |
| **`requires_grad`** | Flag controlling whether a tensor participates in gradient computation. Set to `True` for trainable parameters. |
| **`no_grad()` / `inference_mode()`** | Context managers disabling gradient tracking for efficiency during inference. `inference_mode()` is faster but less flexible. |
| **Custom Autograd Functions** | Inherit from `torch.autograd.Function`, implement `forward()` and `backward()` for operations not in PyTorch or needing custom gradients. |
| **Gradient Accumulation** | `.backward()` accumulates gradients by default (doesn't overwrite). Must call `optimizer.zero_grad()` before each training step. |

### Mental Model: The Complete Picture

```
Training Loop:
1. optimizer.zero_grad()        Clear old gradients
   ↓
2. output = model(input)        Forward pass (graph built dynamically)
   ↓
3. loss = criterion(output)     Compute scalar loss
   ↓
4. loss.backward()              Backward pass (gradients computed)
   ↓
5. optimizer.step()             Update parameters using gradients
   ↓
   (Repeat)

Evaluation:
model.eval()
with torch.inference_mode():    Disable gradients for efficiency
    predictions = model(data)
```

### Design Philosophy: Why Autograd Works This Way

PyTorch's autograd makes specific design choices that reflect its philosophy:

1. **Dynamic over static**: Flexibility and debuggability trump marginal performance gains
2. **Eager execution**: Python runs as you write it; no separate compilation phase
3. **Explicit gradient zeroing**: Accumulation default supports advanced patterns; simple to zero when needed
4. **Leaves store gradients**: Memory-efficient; only parameters need gradient storage
5. **Graph freed after backward**: Memory-efficient default; can retain when needed

### Troubleshooting Checklist

When training isn't working, check these in order:

- ✓ **Are gradients being zeroed?** `optimizer.zero_grad()` before each step?
- ✓ **Is `requires_grad=True`?** On all parameters you want to train?
- ✓ **Is the graph connected?** No accidental `.detach()` or `no_grad()`?
- ✓ **Are gradients flowing?** Check with hooks or print `param.grad` after `.backward()`
- ✓ **Are gradients NaN/Inf?** Enable `detect_anomaly()`, check for numerical issues
- ✓ **Are gradients vanishing/exploding?** Check magnitudes, consider gradient clipping, better initialization, or architecture changes
- ✓ **Is loss actually decreasing?** If not, may be learning rate, initialization, or data issues rather than gradient issues

### Performance Tips

1. **Use `inference_mode()` not `no_grad()` for pure inference** - it's faster
2. **Don't call `.backward()` during evaluation** - wastes computation even with `no_grad()`
3. **Use gradient checkpointing for memory-bound workloads** - trades compute for memory
4. **Batch operations** - compute gradients for batches, not individual samples
5. **Mixed precision** - use `torch.cuda.amp` for faster training with float16 (covered in training guide)

---

**Previous**: [01_tensors.md](01_tensors.md) - Understanding tensor fundamentals and operations
**Next**: [03_modules.md](03_modules.md) - Building neural networks with `nn.Module`

**Related**:
- [Quick Reference](quick_reference.md) - Autograd API cheat sheet
- [Appendix: Debugging Guide](appendix_debugging_guide.md) - Systematic debugging approaches for gradient issues
