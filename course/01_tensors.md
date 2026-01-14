# Tensors: The Foundation of PyTorch

> **Core Question**: How does PyTorch represent and manipulate multidimensional numerical data?

**In this guide:**
- [Overview](#overview)
- [What is a Tensor?](#what-is-a-tensor)
- [Memory Model](#memory-model-the-view-under-the-hood)
- [Broadcasting](#broadcasting-implicit-shape-expansion)
- [Operations](#tensor-operations)
- [Device Management](#device-management-cpugpu)
- [Common Patterns](#common-patterns-and-idioms)
- [Pitfalls](#common-pitfalls)
- [Performance](#performance-considerations)

---

## Overview

A **tensor** is PyTorch's fundamental data structure—a multidimensional array that can:
- Live on CPU or GPU
- Track gradients for automatic differentiation
- Interoperate seamlessly with NumPy

Think of tensors as "NumPy arrays on steroids": they support the same mathematical operations but add GPU acceleration and automatic gradient computation.

### Why Tensors?

1. **Unified representation**: Images, text embeddings, model weights—all tensors
2. **Hardware acceleration**: Single API for CPU and GPU
3. **Automatic differentiation**: Foundation for training neural networks
4. **Optimized operations**: Highly tuned linear algebra kernels

---

## What is a Tensor?

A tensor is characterized by five properties:

### 1. **Data** (the numbers)
```python
tensor([[1.0, 2.0],
        [3.0, 4.0]])
```

### 2. **Shape** (dimensions)
```python
torch.Size([2, 2])  # 2 rows, 2 columns
```

**Terminology**:
- **Scalar**: 0-D tensor, shape `()`
- **Vector**: 1-D tensor, shape `(n,)`
- **Matrix**: 2-D tensor, shape `(m, n)`
- **3D+**: Higher-dimensional tensors, shape `(d1, d2, d3, ...)`

### 3. **Data Type** (dtype)
```python
torch.float32  # Default for most operations
torch.int64    # Default for indexing
torch.bool     # Boolean masks
```

Common dtypes:
| Type | PyTorch | NumPy Equivalent | Use Case |
|------|---------|------------------|----------|
| 32-bit float | `torch.float32` | `np.float32` | **Most neural networks** |
| 64-bit float | `torch.float64` | `np.float64` | High-precision computation |
| 16-bit float | `torch.float16` | `np.float16` | Mixed-precision training |
| 32-bit int | `torch.int32` | `np.int32` | Indexing |
| 64-bit int | `torch.int64` | `np.int64` | **Default for indices** |

### 4. **Device** (CPU or GPU)
```python
torch.device('cpu')
torch.device('cuda:0')  # GPU 0
```

### 5. **Layout** (memory organization)
```python
torch.strided  # Default: row-major order
```

---

## Memory Model: The View Under the Hood

Understanding how tensors are stored in memory is crucial for:
- Performance optimization
- Avoiding unnecessary copies
- Debugging shape errors

### Contiguous Memory

Conceptually, a 2D tensor looks like:
```
[[1, 2, 3],
 [4, 5, 6]]
```

But in memory, it's a **flat 1D array**:
```
Memory: [1][2][3][4][5][6]
```

### Strides: Navigation Instructions

**Strides** tell PyTorch how to jump through memory:

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

x.stride()  # (3, 1)
```

**Interpretation**:
- Stride `(3, 1)` means:
  - To move to the next **row**: skip 3 elements
  - To move to the next **column**: skip 1 element

**Example**:
```
To access x[1, 2] (value 6):
  Start at index 0
  Move 1 row:  0 + (1 × 3) = 3
  Move 2 cols: 3 + (2 × 1) = 5
  Memory[5] = 6 ✓
```

### Contiguous vs Non-Contiguous

A tensor is **contiguous** if elements are laid out in memory in the order you'd traverse them naturally.

```python
x = torch.tensor([[1, 2], [3, 4]])
print(x.is_contiguous())  # True

y = x.t()  # Transpose
print(y.is_contiguous())  # False!
```

After transpose:
```
Logical view:           Memory layout:
[[1, 3],               [1][2][3][4]
 [2, 4]]               ↑     ↑
                       |_____|  Non-sequential!
```

### Views vs Copies

**View**: Same underlying data, different interpretation
```python
x = torch.arange(6)
y = x.view(2, 3)  # View: no data copied

y[0, 0] = 999
print(x)  # tensor([999, 1, 2, 3, 4, 5]) - changed!
```

**Copy**: New independent data
```python
x = torch.arange(6)
y = x.clone().view(2, 3)  # Copy first

y[0, 0] = 999
print(x)  # tensor([0, 1, 2, 3, 4, 5]) - unchanged
```

**Key operations that return views**:
- `view()`, `reshape()` (when possible)
- `transpose()`, `t()`
- `narrow()`, `select()`
- Basic indexing: `x[0]`, `x[:, 1]`

**Operations that return copies**:
- `clone()`
- `contiguous()`
- Fancy indexing: `x[[0, 2]]`
- Operations like `+`, `*` (create new tensors)

---

## Broadcasting: Implicit Shape Expansion

Broadcasting allows operations on tensors of different shapes without explicit replication.

### The Three Rules

1. **Right-align shapes**
2. **Dimensions of size 1 can stretch** to match the other tensor
3. **Missing dimensions** are treated as size 1

### Example 1: Vector + Scalar

```python
x = torch.tensor([1, 2, 3])  # Shape: (3,)
y = 10                        # Shape: ()  (scalar)

# Broadcasting:
x:        [1, 2, 3]  # (3,)
y:        10         # () → (1,) → (3,) stretched
result:   [11, 12, 13]
```

### Example 2: Matrix + Vector

```python
x = torch.randn(3, 4)  # (3, 4)
y = torch.randn(4)     # (4,)

# Broadcasting:
x:        (3, 4)
y:        (   4) → (1, 4) → (3, 4) stretched
result:   (3, 4)
```

### Example 3: 3D Tensor Operations

```python
x = torch.randn(8, 1, 6)  # (8, 1, 6)
y = torch.randn(   7, 1)  # (   7, 1)

# Right-align:
x:        (8, 1, 6)
y:        (   7, 1)

# Stretch size-1 dims:
x →       (8, 7, 6)  # dim 1: 1 → 7
y →       (8, 7, 1) → (8, 7, 6)  # dim 0: missing → 8, dim 2: 1 → 6

result:   (8, 7, 6)
```

### When Broadcasting Fails

```python
x = torch.randn(3, 4)
y = torch.randn(3, 5)

# Cannot broadcast (3, 4) with (3, 5)
#                    ↑           ↑
#                    4 ≠ 5 and neither is 1
```

### Explicit Broadcasting

Sometimes you want to be explicit:

```python
x = torch.randn(3, 1)
y = torch.randn(1, 4)

# Option 1: Let broadcasting happen
result = x + y  # (3, 4)

# Option 2: Explicit with expand
x_expanded = x.expand(3, 4)  # View, no memory allocation
y_expanded = y.expand(3, 4)
result = x_expanded + y_expanded
```

---

## Tensor Operations

### Creation

```python
# From data
torch.tensor([1, 2, 3])                    # From Python list
torch.as_tensor(numpy_array)               # From NumPy (shares memory!)
torch.from_numpy(numpy_array)              # From NumPy (shares memory!)

# Zeros and ones
torch.zeros(3, 4)                          # All zeros
torch.ones(2, 3)                           # All ones
torch.full((2, 3), 7.0)                    # Filled with value
torch.eye(3)                               # Identity matrix

# Random initialization
torch.rand(2, 3)                           # Uniform [0, 1)
torch.randn(2, 3)                          # Normal N(0, 1)
torch.randint(0, 10, (3, 4))              # Random integers

# Ranges
torch.arange(0, 10, 2)                    # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)                   # [0.0, 0.25, 0.5, 0.75, 1.0]

# Like another tensor
x = torch.randn(3, 4)
torch.zeros_like(x)                        # Same shape, dtype, device
torch.ones_like(x)
torch.randn_like(x)
```

### Shape Manipulation

```python
x = torch.randn(2, 3, 4)

# Reshape
x.view(6, 4)                  # Requires contiguous tensor
x.reshape(6, 4)               # Copies if needed
x.view(-1, 4)                 # Infer dimension: -1 → 6

# Squeeze/Unsqueeze
x = torch.randn(1, 3, 1, 4)
x.squeeze()                   # Remove all size-1 dims → (3, 4)
x.squeeze(0)                  # Remove specific dim → (3, 1, 4)

y = torch.randn(3, 4)
y.unsqueeze(0)                # Add dim at position 0 → (1, 3, 4)
y.unsqueeze(-1)               # Add dim at end → (3, 4, 1)

# Transpose
x = torch.randn(2, 3)
x.t()                         # Transpose 2D: (3, 2)
x.transpose(0, 1)             # Swap dims 0 and 1

x = torch.randn(2, 3, 4)
x.permute(2, 0, 1)           # Reorder dims → (4, 2, 3)

# Flatten
x = torch.randn(2, 3, 4)
x.flatten()                   # (24,)
x.flatten(start_dim=1)        # (2, 12) - keep first dim
```

### Indexing and Slicing

```python
x = torch.randn(4, 5)

# Basic indexing
x[0]                          # First row: (5,)
x[:, 1]                       # Second column: (4,)
x[1:3, 2:4]                   # Submatrix: (2, 2)

# Advanced indexing
x[[0, 2]]                     # Select rows 0 and 2: (2, 5)
x[:, [1, 3]]                  # Select columns 1 and 3: (4, 2)

# Boolean masking
mask = x > 0
positive = x[mask]            # All positive values (1D tensor)

# torch.where (ternary operator)
torch.where(x > 0, x, torch.zeros_like(x))  # ReLU-like
```

### Mathematical Operations

```python
# Element-wise
x + y, x - y, x * y, x / y
x.add(y), x.sub(y), x.mul(y), x.div(y)
x.pow(2), x.sqrt(), x.exp(), x.log()
torch.sin(x), torch.cos(x)

# Linear algebra
torch.matmul(x, y)            # Matrix multiplication
x @ y                         # Same as matmul
torch.mm(x, y)                # 2D matmul
torch.bmm(x, y)               # Batched matmul
x.dot(y)                      # Dot product (1D)

# Reductions
x.sum(), x.mean(), x.std()
x.max(), x.min()
x.argmax(), x.argmin()
x.sum(dim=0)                  # Sum along dimension 0
x.mean(dim=1, keepdim=True)   # Keep reduced dimension
```

### In-Place Operations

Operations with `_` suffix modify the tensor in-place:

```python
x = torch.randn(3, 4)

x.add_(5)        # x += 5 (in-place)
x.mul_(2)        # x *= 2 (in-place)
x.clamp_(0, 1)   # Clamp values (in-place)
```

⚠️ **Warning**: Be careful with in-place ops on tensors that require gradients!

---

## Device Management (CPU/GPU)

### Moving Tensors

```python
x = torch.randn(3, 4)

# Move to GPU
x_gpu = x.to('cuda')
x_gpu = x.cuda()              # Shorthand

# Move back to CPU
x_cpu = x_gpu.to('cpu')
x_cpu = x_gpu.cpu()           # Shorthand

# Move to specific GPU
x_gpu0 = x.to('cuda:0')
x_gpu1 = x.to('cuda:1')
```

### Best Practice: Device-Agnostic Code

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(3, 4, device=device)  # Created on device
y = torch.randn(3, 4).to(device)      # Moved to device
```

### Memory Management

```python
# Check GPU memory
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()

# Clear cache
torch.cuda.empty_cache()

# Context manager for automatic memory management
with torch.cuda.device(0):
    x = torch.randn(1000, 1000, device='cuda')
```

---

## Common Patterns and Idioms

### Pattern 1: Batch Operations

```python
# Single sample: (channels, height, width)
image = torch.randn(3, 224, 224)

# Batch: (batch_size, channels, height, width)
batch = torch.randn(32, 3, 224, 224)

# Add batch dimension
image_batch = image.unsqueeze(0)  # (1, 3, 224, 224)
```

### Pattern 2: One-Hot Encoding

```python
labels = torch.tensor([0, 2, 1])  # Class indices
num_classes = 3

one_hot = F.one_hot(labels, num_classes)
# tensor([[1, 0, 0],
#         [0, 0, 1],
#         [0, 1, 0]])
```

### Pattern 3: Masking

```python
x = torch.randn(100)

# Create mask
mask = x > 0

# Apply mask
positive_values = x[mask]

# Conditional replacement
x = torch.where(mask, x, torch.zeros_like(x))  # Keep positive, zero negative
```

### Pattern 4: Concatenation and Stacking

```python
x = torch.randn(3, 4)
y = torch.randn(3, 4)

# Concatenate along existing dimension
torch.cat([x, y], dim=0)     # (6, 4)
torch.cat([x, y], dim=1)     # (3, 8)

# Stack creates new dimension
torch.stack([x, y], dim=0)   # (2, 3, 4)
torch.stack([x, y], dim=1)   # (3, 2, 4)
```

---

## Common Pitfalls

### Pitfall 1: Unintended Broadcasting

```python
# Intended: element-wise multiply
x = torch.randn(10, 1)
y = torch.randn(10)

result = x * y  # Result is (10, 10), not (10, 1)!
```

**Solution**: Be explicit about shapes
```python
result = x * y.unsqueeze(1)  # (10, 1) * (10, 1) = (10, 1)
```

### Pitfall 2: In-Place Operations on Leaves

```python
x = torch.randn(3, requires_grad=True)
x += 1  # RuntimeError: leaf variable requires grad
```

**Solution**: Use out-of-place operations
```python
x = x + 1  # Creates new tensor
```

### Pitfall 3: View on Non-Contiguous Tensor

```python
x = torch.randn(3, 4).t()  # Transpose → non-contiguous
y = x.view(2, 6)           # RuntimeError!
```

**Solution**: Make contiguous first
```python
y = x.contiguous().view(2, 6)
# Or use reshape (copies if needed)
y = x.reshape(2, 6)
```

### Pitfall 4: Device Mismatch

```python
x = torch.randn(3, 4, device='cuda')
y = torch.randn(3, 4)  # Default: CPU

z = x + y  # RuntimeError: tensors on different devices
```

**Solution**: Ensure same device
```python
z = x + y.to(x.device)
```

---

## Performance Considerations

### 1. Vectorize Operations

❌ **Slow**: Loops
```python
result = []
for i in range(len(x)):
    result.append(x[i] * 2)
result = torch.stack(result)
```

✅ **Fast**: Vectorized
```python
result = x * 2
```

### 2. Avoid Unnecessary Copies

```python
# View (fast, no copy)
y = x.view(2, -1)

# Reshape (may copy if non-contiguous)
y = x.reshape(2, -1)

# Explicit copy
y = x.clone()
```

### 3. In-Place When Safe

```python
# Creates new tensor
x = x + 1

# In-place (saves memory)
x += 1  # or x.add_(1)
```

### 4. GPU Memory Transfer is Expensive

```python
# Bad: Many small transfers
for i in range(1000):
    x_gpu = x[i].cuda()  # Transfer each element

# Good: Single batch transfer
x_gpu = x.cuda()  # Transfer once
for i in range(1000):
    use(x_gpu[i])
```

### 5. Use `.to(device, non_blocking=True)` for Async Transfer

```python
# Synchronous (blocks CPU)
x_gpu = x.to('cuda')

# Asynchronous (overlaps with CPU work)
x_gpu = x.to('cuda', non_blocking=True)
```

---

## Summary

### Key Takeaways

| Concept | Essential Points |
|---------|-----------------|
| **Tensors** | Multidimensional arrays with dtype, shape, and device |
| **Memory** | Stored as flat arrays; strides define navigation; views share data |
| **Broadcasting** | Automatic shape expansion following three rules |
| **Devices** | CPU vs GPU; use device-agnostic code |
| **Performance** | Vectorize, avoid copies, minimize device transfers |

### Quick Checklist

Before operations, ask:
- ✓ Are shapes compatible (broadcasting)?
- ✓ Are tensors on the same device?
- ✓ Is the tensor contiguous (for `.view()`)?
- ✓ Will this create a copy or a view?
- ✓ Do I need gradients (for autograd)?

---

**Next**: [02_autograd.md](02_autograd.md) - Learn how PyTorch computes gradients automatically

**Related**:
- [Quick Reference](quick_reference.md) - Tensor operations cheat sheet
- [Appendix: Memory Deep Dive](appendix_memory_deep_dive.md) - Advanced memory topics
