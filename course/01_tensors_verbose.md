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

A **tensor** is PyTorch's fundamental data structure—a multidimensional array that forms the foundation of every computation in the framework. Think of tensors as generalized matrices that can represent data in any number of dimensions, from simple scalars (zero dimensions) to vectors (one dimension), matrices (two dimensions), and beyond into higher-dimensional structures that are essential for modern deep learning. Every piece of data you work with in PyTorch, whether it's an image, a text embedding, a batch of training examples, or the weights of a neural network, is ultimately represented as a tensor.

What makes PyTorch tensors particularly powerful is that they can live on different computational devices (CPU or GPU), track gradients for automatic differentiation during training, and interoperate seamlessly with NumPy arrays when you need to work with the broader Python scientific computing ecosystem. In many ways, tensors are like "NumPy arrays on steroids"—they support all the same mathematical operations you're familiar with from NumPy, but they add crucial capabilities like GPU acceleration for massive parallelism and automatic gradient computation that forms the backbone of neural network training.

### Why Tensors?

The tensor abstraction is central to PyTorch for several deeply interconnected reasons that reflect fundamental requirements of modern machine learning:

1. **Unified representation across data types**: Whether you're working with images (typically represented as 3D tensors with dimensions for height, width, and color channels), text embeddings (2D tensors with dimensions for sequence length and embedding size), or model weights (which might be 2D for linear layers or 4D for convolutional filters), everything is a tensor. This unified representation means you can use the same operations, the same optimization techniques, and the same mental model across wildly different types of data. When a convolutional layer processes an image and a recurrent layer processes text, they're both fundamentally performing mathematical operations on tensors, which allows PyTorch to provide a consistent, composable API.

2. **Hardware acceleration with a single API**: Modern deep learning requires enormous computational throughput, which is why GPUs (and increasingly, specialized accelerators like TPUs) have become essential. Tensors provide a single, unified interface for performing computations on different hardware backends. You can write code that operates on tensors, and with a single `.cuda()` call, move those tensors to the GPU where the same operations will run orders of magnitude faster through parallel processing. This abstraction over hardware means you don't need to rewrite your algorithms when you move from prototyping on a laptop to training on a GPU cluster.

3. **Automatic differentiation foundation**: Neural networks learn through backpropagation, which requires computing gradients of loss functions with respect to model parameters. PyTorch's autograd system (which we'll explore in depth in the next guide) builds computation graphs using tensors as nodes. Every operation you perform on tensors can potentially be differentiated automatically, allowing the framework to compute gradients without you ever having to derive or implement the calculus manually. This capability transforms what would be error-prone, tedious mathematical programming into simple, intuitive code.

4. **Optimized operations through specialized kernels**: Behind PyTorch's simple API lie highly tuned linear algebra libraries like BLAS, cuBLAS, and cuDNN that have been optimized over decades. When you multiply two tensors together, you're not running a naive triple-nested loop; you're invoking carefully crafted algorithms that exploit cache hierarchies, SIMD instructions, and GPU parallelism. The tensor abstraction allows PyTorch to dispatch your high-level operations to these low-level, optimized kernels automatically.

---

## What is a Tensor?

A tensor in PyTorch is fundamentally characterized by five essential properties that together completely define its identity and behavior. Understanding these properties is crucial because they determine what operations you can perform, how efficiently those operations will run, and how the tensor interacts with other tensors in your computations.

### 1. **Data** (the actual numerical values)

At its core, a tensor contains actual numerical data—the raw values you're working with. This might be pixel intensities in an image, word embedding coefficients, or the learned weights of a neural network layer:

```python
tensor([[1.0, 2.0],
        [3.0, 4.0]])
```

These are the values you'll manipulate with mathematical operations, and they're stored contiguously in memory for efficient access (we'll explore the memory layout in detail shortly).

### 2. **Shape** (the dimensional structure)

The **shape** describes the tensor's dimensional structure—how many dimensions it has and the size of each dimension. In PyTorch, shapes are represented as `torch.Size` objects, which behave like tuples:

```python
torch.Size([2, 2])  # 2 rows, 2 columns
```

The terminology for tensors of different dimensionalities is borrowed from mathematics and physics, and it's worth understanding precisely what each term means:

- **Scalar**: A 0-dimensional tensor with shape `()`. This represents a single number with no dimensions at all. While it might seem odd to call a single number a "tensor," this consistent abstraction allows PyTorch to treat all data uniformly. When you compute a loss value, for example, you get back a scalar tensor.

- **Vector**: A 1-dimensional tensor with shape `(n,)`. This is like a Python list or NumPy array—a single sequence of numbers indexed by one position. Vectors represent things like a single sequence of word embeddings or a row/column of pixels.

- **Matrix**: A 2-dimensional tensor with shape `(m, n)`. This is the familiar structure from linear algebra, with rows and columns. Weight matrices in fully connected layers, grayscale images, and many data tables are naturally represented as matrices.

- **3D and higher**: Tensors with three or more dimensions, like shape `(d1, d2, d3, ...)`. These higher-dimensional structures are ubiquitous in deep learning. For example, a batch of RGB images has four dimensions: (batch_size, channels, height, width). The ability to work seamlessly with these higher-dimensional structures is what separates tensor libraries from traditional matrix libraries.

### 3. **Data Type** (dtype - the numeric precision)

The **dtype** specifies how each number in the tensor is represented in memory—whether it's a 32-bit floating-point number, a 64-bit integer, a boolean, and so on. This matters enormously for both memory efficiency and numerical precision:

```python
torch.float32  # Default for most neural network operations
torch.int64    # Default for indexing and integer operations
torch.bool     # Boolean masks and logical operations
```

Understanding dtypes is crucial because choosing the wrong dtype can either waste memory (using `float64` when `float32` suffices) or cause numerical errors (using integers for operations that need fractional values). Here's a breakdown of the most common dtypes and when you'd use each:

| Type | PyTorch Name | NumPy Equivalent | Typical Use Case | Notes |
|------|--------------|------------------|------------------|-------|
| 32-bit float | `torch.float32` (or `torch.float`) | `np.float32` | **Most neural networks** | The standard choice: good precision, reasonable memory, hardware-accelerated |
| 64-bit float | `torch.float64` (or `torch.double`) | `np.float64` | High-precision scientific computing | Double the memory of float32; use when numerical precision is critical |
| 16-bit float | `torch.float16` (or `torch.half`) | `np.float16` | Mixed-precision training on GPUs | Saves memory and speeds up computation but with reduced precision |
| 32-bit int | `torch.int32` (or `torch.int`) | `np.int32` | Integer computations, some indexing | Standard integer type |
| 64-bit int | `torch.int64` (or `torch.long`) | `np.int64` | **Indexing, class labels** | PyTorch's default integer type for indices |
| 8-bit int | `torch.uint8` | `np.uint8` | Image data (pixel values 0-255) | Saves memory for image storage |
| Boolean | `torch.bool` | `np.bool_` | Masks, logical operations | Single bit per value (in principle) |

The choice of dtype represents a fundamental tradeoff in computing: precision versus efficiency. Most neural networks use **float32** because it provides sufficient numerical precision for gradient-based optimization while being well-supported by GPU hardware. Using **float64** would double your memory consumption and slow down training, typically without meaningful improvements in model performance. Conversely, **float16** (half precision) is increasingly popular for large models because it halves memory requirements and can dramatically speed up training on modern GPUs with specialized tensor cores, though it requires careful handling to avoid numerical instability.

### 4. **Device** (where the tensor lives - CPU or GPU)

The **device** property specifies which computational hardware the tensor's data resides on. This is a crucial concept because tensors can only interact with other tensors on the same device, and moving data between devices has significant performance implications:

```python
torch.device('cpu')      # Tensor lives in main system RAM
torch.device('cuda:0')   # Tensor lives in GPU 0's memory
torch.device('cuda:1')   # Tensor lives in GPU 1's memory (multi-GPU systems)
```

The device abstraction is what enables PyTorch to seamlessly leverage GPUs for acceleration. When you move a tensor to a CUDA device (the name for NVIDIA GPUs in PyTorch), subsequent operations on that tensor will be dispatched to GPU kernels that can process thousands of elements in parallel. However, this power comes with the responsibility of managing device placement—you cannot add a CPU tensor to a GPU tensor directly, and moving data between CPU and GPU memory is one of the most common performance bottlenecks in deep learning applications.

### 5. **Layout** (how data is organized in memory)

The **layout** describes how the tensor's logical multidimensional structure is mapped onto physical linear memory. For almost all use cases, you'll work with the default **strided** layout, which uses a row-major memory ordering (we'll explore exactly what this means in the next section):

```python
torch.strided  # Default: row-major order with configurable strides
```

While there are other specialized layouts (like `torch.sparse_coo` for sparse tensors where most values are zero), the strided layout is so ubiquitous that many PyTorch users never explicitly think about it. However, understanding how strides work is essential for understanding the difference between views and copies, and for diagnosing performance issues related to memory access patterns.

---

## Memory Model: The View Under the Hood

To truly understand tensors and use them effectively, you need to understand how they're actually stored in computer memory. This understanding is crucial for three practical reasons: it allows you to optimize performance by avoiding unnecessary data copies, it helps you understand why certain operations fail (like trying to `.view()` a non-contiguous tensor), and it enables you to debug subtle bugs related to shared memory when you don't expect it.

### The Fundamental Insight: Flat Memory with Strides

Here's the key insight that underpins all of tensor memory management: although we conceptualize a tensor as a multidimensional array with rows and columns (and potentially many more dimensions), computer memory is fundamentally a one-dimensional array of bytes. A 2D tensor that looks like this conceptually:

```
[[1, 2, 3],
 [4, 5, 6]]
```

...is actually stored in memory as a flat, linear sequence:

```
Memory: [1][2][3][4][5][6]
         ↑                 ↑
      Address 0        Address 5
```

This representation is called **row-major order** (also known as C-style ordering), and it's the default in PyTorch, C, and most programming languages. The alternative, **column-major order** (used by Fortran and MATLAB), would store the columns sequentially: `[1][4][2][5][3][6]`. Row-major order is intuitive for how we typically think about matrices: we store the first row entirely, then the second row, and so on.

### Strides: The Navigation System

So if everything is stored in a flat array, how does PyTorch know how to interpret `tensor[1, 2]` (row 1, column 2) and find the right element in memory? The answer is **strides**—a tuple of integers that serves as a navigation system for the flat memory array.

Consider our 2x3 tensor again:

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.stride())  # (3, 1)
```

The stride tuple `(3, 1)` tells PyTorch how many elements to skip in memory to move one step along each dimension:

- **Stride of 3 for dimension 0 (rows)**: To move from row 0 to row 1, skip 3 elements in memory. This makes sense because each row contains 3 elements, so the start of row 1 is 3 positions after the start of row 0.

- **Stride of 1 for dimension 1 (columns)**: To move from column 0 to column 1 within the same row, skip 1 element in memory. This also makes sense because adjacent columns are adjacent in memory.

To access any element, PyTorch uses this formula:

```
memory_offset = sum(index[i] * stride[i] for i in range(ndim))
```

Let's trace through accessing `x[1, 2]` (which should give us the value 6):

```
Starting at memory address 0:
  Move 1 step in dimension 0 (rows):    0 + (1 × 3) = 3
  Move 2 steps in dimension 1 (columns): 3 + (2 × 1) = 5
  Memory[5] = 6 ✓
```

This stride system is remarkably elegant because it allows PyTorch to represent many different "views" of the same underlying data without copying anything. By changing the stride tuple, you can reinterpret the same flat memory in different ways.

### Contiguous vs Non-Contiguous Tensors

A tensor is called **contiguous** if its elements are stored in memory in the exact order you'd encounter them by iterating through the tensor's logical structure. More precisely, a tensor is contiguous if it's stored in row-major order with no gaps or unusual stride patterns.

```python
x = torch.tensor([[1, 2], [3, 4]])
print(x.is_contiguous())  # True

y = x.t()  # Transpose: swap rows and columns
print(y.is_contiguous())  # False!
```

Why is the transposed tensor non-contiguous? After transposing, the logical view is:

```
[[1, 3],
 [2, 4]]
```

But the memory layout hasn't changed—it's still `[1][2][3][4]`. To traverse the first row of the transposed tensor (1, 3), we'd need to access memory indices 0 and 2, which aren't adjacent. The stride has changed from `(2, 1)` to `(1, 2)`, indicating that moving one row now only skips 1 element (where column steps used to be), while moving one column skips 2 elements (where row steps used to be). This pattern of non-adjacent access makes the tensor non-contiguous.

Why does contiguity matter? Many PyTorch operations (like `.view()`) require contiguous tensors because they rely on the memory being laid out in a predictable pattern. If you try to reshape a non-contiguous tensor with `.view()`, PyTorch will raise an error because it can't guarantee the reshape will have the meaning you expect without actually reorganizing the data in memory. In such cases, you have two options: call `.contiguous()` to create a new contiguous copy, or use `.reshape()` which will copy only if necessary.

### Views vs Copies: The Most Important Distinction

One of the most important concepts in PyTorch tensor management is the difference between **views** and **copies**. Understanding this distinction prevents subtle bugs where modifying one tensor unexpectedly changes another, and it's essential for writing memory-efficient code.

A **view** is a tensor that shares the same underlying memory as another tensor but potentially interprets it differently (through different shape or strides). Creating a view is fast (just updating metadata) and memory-efficient (no data duplication), but it means modifications to one tensor affect the other:

```python
x = torch.arange(6)  # tensor([0, 1, 2, 3, 4, 5])
y = x.view(2, 3)      # View: same memory, different shape

y[0, 0] = 999
print(x)  # tensor([999, 1, 2, 3, 4, 5]) - changed!
```

When you modified `y[0, 0]`, you actually modified the first element of the underlying memory that both `x` and `y` share. This behavior can be surprising if you're not expecting it, but it's also powerful for building memory-efficient pipelines.

A **copy**, on the other hand, creates a completely independent tensor with its own memory allocation. Modifications to the copy don't affect the original:

```python
x = torch.arange(6)
y = x.clone().view(2, 3)  # Clone creates a copy first

y[0, 0] = 999
print(x)  # tensor([0, 1, 2, 3, 4, 5]) - unchanged!
```

**Common operations that return views** (fast, share memory):
- `view()` and `reshape()` (when the tensor is contiguous)
- `transpose()`, `t()`, `permute()` (reorder dimensions)
- `narrow()`, `select()` (select a slice)
- Basic indexing with slices: `x[0]`, `x[:, 1]`, `x[1:3]`
- `expand()` (broadcast dimensions)
- `unsqueeze()` and `squeeze()` (add/remove singleton dimensions)

**Operations that return copies** (slower, independent memory):
- `clone()` (explicit copy)
- `contiguous()` (copy if non-contiguous)
- Fancy indexing: `x[[0, 2]]`, `x[mask]` (non-contiguous selections)
- Arithmetic operations: `x + y`, `x * 2` (create new result tensors)
- `reshape()` when the tensor is non-contiguous

The general principle is: operations that just reinterpret existing memory return views, while operations that need to reorganize or compute new values return copies. When in doubt, you can check by modifying the result and seeing if the original changes, or by using `data_ptr()` to see if two tensors point to the same memory.

---

## Broadcasting: Implicit Shape Expansion

**Broadcasting** is one of PyTorch's most elegant features, and it's also one of the most common sources of confusion for newcomers. Broadcasting allows you to perform operations between tensors of different shapes without explicitly replicating data, making your code more concise and efficient. However, the rules for when and how broadcasting applies can seem mysterious if you don't understand the underlying logic.

The fundamental idea behind broadcasting is this: when you try to perform an element-wise operation (like addition or multiplication) between two tensors of different shapes, PyTorch will automatically "stretch" the smaller tensor to match the larger one's shape—but only in dimensions where this makes sense, and only conceptually (no actual data copying happens).

### The Three Broadcasting Rules

Broadcasting follows three rules that are applied in sequence:

1. **Right-align the shapes**: Compare the shapes starting from the trailing (rightmost) dimensions and work backward. If one tensor has fewer dimensions than the other, mentally prepend dimensions of size 1 to the left of its shape.

2. **Dimensions of size 1 can stretch**: In any dimension, if one tensor has size 1 and the other has size n, the size-1 dimension is conceptually stretched to size n by repeating the values.

3. **Missing dimensions are treated as size 1**: If one tensor has fewer dimensions than the other, the missing dimensions (on the left) are treated as if they were present with size 1, and then rule 2 applies.

If after applying these rules the shapes still don't match, broadcasting fails and PyTorch raises an error. Let's work through examples to make this concrete.

### Example 1: Vector + Scalar

```python
x = torch.tensor([1, 2, 3])  # Shape: (3,)
y = 10                        # Shape: () - scalar

result = x + y  # Broadcasting happens automatically
# Result: tensor([11, 12, 13])
```

**What happened?**
1. Right-align: `x` is (3,) and `y` is () (0 dimensions)
2. Prepend to `y`: () → (1,) conceptually
3. Stretch: (1,) → (3,) to match `x`
4. Now both are conceptually (3,), so element-wise addition proceeds

The scalar 10 is conceptually replicated three times, but PyTorch doesn't actually create three copies in memory—it just accesses the same value three times.

### Example 2: Matrix + Vector (Row-wise Operation)

```python
x = torch.randn(3, 4)  # Shape: (3, 4) - 3 rows, 4 columns
y = torch.randn(4)     # Shape: (4,) - single row

result = x + y  # Add y to each row of x
# Result shape: (3, 4)
```

**What happened?**
1. Right-align shapes:
   ```
   x:    (3, 4)
   y:       (4)
   ```
2. Prepend to `y`: (4) → (1, 4)
3. Stretch dimension 0 of `y`: (1, 4) → (3, 4)
4. Now both are (3, 4), so addition proceeds

This is an incredibly common pattern in neural networks: adding a bias vector to each row of a batch of activations.

### Example 3: 3D Tensor Broadcasting

```python
x = torch.randn(8, 1, 6)  # Shape: (8, 1, 6)
y = torch.randn(   7, 1)  # Shape: (   7, 1)

result = x + y  # What's the result shape?
# Result shape: (8, 7, 6)
```

**What happened?**
1. Right-align:
   ```
   x:    (8, 1, 6)
   y:       (7, 1)
   ```
2. Prepend to `y`: (7, 1) → (1, 7, 1) (adding a dimension on the left)
3. Now compare dimension by dimension:
   - Dimension 0: 8 vs 1 → stretch `y` to 8
   - Dimension 1: 1 vs 7 → stretch `x` to 7
   - Dimension 2: 6 vs 1 → stretch `y` to 6
4. Result: both conceptually (8, 7, 6)

This example shows how broadcasting can expand multiple dimensions simultaneously, which is powerful but can be surprising if you're not expecting it.

### When Broadcasting Fails

Broadcasting fails when dimensions are incompatible—meaning they're different sizes and neither is 1:

```python
x = torch.randn(3, 4)
y = torch.randn(3, 5)

# Attempting: x + y
# Compare: (3, 4) vs (3, 5)
#               ↑         ↑
#               4 ≠ 5 and neither is 1
# Error: RuntimeError: The size of tensor a (4) must match the size of tensor b (5)
```

### Explicit Broadcasting with `expand()` and `expand_as()`

While implicit broadcasting is convenient, sometimes you want to be explicit about what's happening, or you need to create a view with a broadcasted shape for some operation. That's where `expand()` comes in:

```python
x = torch.randn(3, 1)
y = torch.randn(1, 4)

# Implicit broadcasting (preferred for operations)
result = x + y  # (3, 4)

# Explicit broadcasting (useful for inspection or special cases)
x_expanded = x.expand(3, 4)  # View with shape (3, 4), no memory allocation
y_expanded = y.expand(3, 4)  # View with shape (3, 4), no memory allocation
result = x_expanded + y_expanded  # (3, 4)
```

It's important to note that `expand()` returns a view, not a copy—it's just metadata manipulation. You can also use `expand_as(other)` to expand a tensor to match another tensor's shape.

### Why Broadcasting Matters

Broadcasting is not just a convenience feature—it's fundamental to writing efficient, idiomatic PyTorch code. Without broadcasting, you'd need to manually replicate data with operations like `repeat()` or `tile()`, which would create actual copies in memory and slow down your code. Broadcasting lets you write concise expressions like `x + bias` instead of verbose, inefficient code that manually tiles the bias across all dimensions. It's also essential for batched operations in neural networks, where you routinely need to apply the same operation to every element in a batch.

---

## Tensor Operations

PyTorch provides a vast library of operations for creating, manipulating, and computing with tensors. These operations form the building blocks of everything you'll do in deep learning, from loading data to implementing custom layers to training models. Let's survey the most important categories of operations and understand when and why you'd use each.

### Creation Operations: Bringing Tensors Into Existence

Every tensor has to come from somewhere. PyTorch provides numerous ways to create tensors depending on whether you're starting from existing data, need initialized values for model parameters, want random values for testing, or need specific patterns:

**From existing data:**
```python
# From Python lists or tuples
torch.tensor([1, 2, 3])
torch.tensor([[1, 2], [3, 4]])

# From NumPy arrays (shares memory! - be careful)
torch.as_tensor(numpy_array)
torch.from_numpy(numpy_array)

# From NumPy with explicit copy
torch.tensor(numpy_array)
```

An important subtlety: `torch.from_numpy()` and `torch.as_tensor()` share memory with the NumPy array when possible, meaning modifications to the tensor will affect the original NumPy array and vice versa. This is efficient but can be surprising. If you want an independent copy, use `torch.tensor()` instead.

**Zeros, ones, and constant fills:**
```python
torch.zeros(3, 4)           # 3x4 matrix of zeros (default float32)
torch.ones(2, 3)            # 2x3 matrix of ones
torch.full((2, 3), 7.0)     # 2x3 matrix filled with 7.0
torch.eye(3)                # 3x3 identity matrix (diagonal of ones)
```

These operations are fundamental for initializing accumulators, creating masks, and initializing certain types of model parameters.

**Random initialization:**
```python
torch.rand(2, 3)            # Uniform distribution on [0, 1)
torch.randn(2, 3)           # Standard normal distribution N(0, 1)
torch.randint(0, 10, (3, 4)) # Random integers from [0, 10)
```

Random initialization is crucial for neural network weights. Most modern networks initialize weights with `randn()` (normal distribution) scaled by a factor that depends on layer size (Xavier or He initialization) to ensure stable gradient flow during early training.

**Ranges and sequences:**
```python
torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8] - like Python range()
torch.linspace(0, 1, 5)     # [0.0, 0.25, 0.5, 0.75, 1.0] - evenly spaced
```

These are useful for creating coordinate grids, positional encodings, and test data.

**Creating tensors like existing tensors:**
```python
x = torch.randn(3, 4)
torch.zeros_like(x)         # Same shape, dtype, device as x, but filled with zeros
torch.ones_like(x)
torch.randn_like(x)
```

The `_like()` variants are extremely convenient when you need to create a new tensor that matches an existing tensor's properties—common when accumulating gradients or building masks.

### Shape Manipulation: Reshaping Your Data

Neural network architectures often require you to reshape data as it flows through layers. Understanding the subtleties of reshape operations is essential:

**Reshaping:**
```python
x = torch.randn(2, 3, 4)  # 24 total elements

x.view(6, 4)              # Reshape to 6x4 (requires contiguous memory)
x.reshape(6, 4)           # Reshape to 6x4 (copies if non-contiguous)
x.view(-1, 4)             # Infer dimension: -1 automatically computes 6
```

The difference between `view()` and `reshape()` is subtle but important: `view()` returns a view of the existing memory and will error if the tensor isn't contiguous, while `reshape()` will copy if necessary to make the reshape work. For maximum performance, prefer `view()` when you know your tensor is contiguous; for maximum flexibility, use `reshape()`.

**Squeeze and Unsqueeze:**
```python
x = torch.randn(1, 3, 1, 4)
x.squeeze()               # Remove all size-1 dims → (3, 4)
x.squeeze(0)              # Remove only dimension 0 → (3, 1, 4)
x.squeeze(2)              # Remove only dimension 2 → (1, 3, 4)

y = torch.randn(3, 4)
y.unsqueeze(0)            # Add dimension at position 0 → (1, 3, 4)
y.unsqueeze(-1)           # Add dimension at end → (3, 4, 1)
y[None, :, :]             # Equivalent to unsqueeze(0)
```

These operations are essential for adding batch dimensions (unsqueezing) or removing them after processing (squeezing). The ability to use `-1` or `None` for dimension indices makes this code more readable and robust.

**Transpose and Permute:**
```python
x = torch.randn(2, 3)
x.t()                     # Transpose 2D tensor: (3, 2)
x.transpose(0, 1)         # Swap dimensions 0 and 1 (works for any rank)

x = torch.randn(2, 3, 4)
x.permute(2, 0, 1)        # Reorder to (4, 2, 3)
```

Transposing is common when you need to convert between different data layouts (like switching from batch-first to sequence-first for RNNs).

**Flatten:**
```python
x = torch.randn(2, 3, 4)
x.flatten()               # Flatten to 1D: (24,)
x.flatten(start_dim=1)    # Flatten all but first dim: (2, 12)
```

Flattening is ubiquitous when transitioning from convolutional layers (which expect 4D tensors) to fully connected layers (which expect 2D tensors).

### Indexing and Slicing: Selecting Subsets

**Basic indexing with slices** (returns views):
```python
x = torch.randn(4, 5)

x[0]                      # First row: (5,)
x[:, 1]                   # Second column: (4,)
x[1:3, 2:4]              # Submatrix: (2, 2)
x[:-1]                    # All but last row: (3, 5)
```

**Advanced indexing** (returns copies):
```python
x[[0, 2]]                 # Select rows 0 and 2: (2, 5)
x[:, [1, 3]]              # Select columns 1 and 3: (4, 2)
```

**Boolean masking** (incredibly powerful for conditional logic):
```python
mask = x > 0
positive = x[mask]        # All positive values (1D tensor)

# Conditional replacement
x[x < 0] = 0              # Set all negative values to zero (ReLU-like)

# torch.where: ternary operator (condition ? x : y)
torch.where(x > 0, x, torch.zeros_like(x))
```

Boolean masking is one of the most important patterns in PyTorch because it lets you express conditional logic in a vectorized way without Python loops.

### Mathematical Operations: The Core of Computation

**Element-wise arithmetic** (broadcasting applies):
```python
x + y, x - y, x * y, x / y
x.add(y), x.sub(y), x.mul(y), x.div(y)
x.pow(2), x.sqrt(), x.exp(), x.log()
torch.sin(x), torch.cos(x), torch.tanh(x)
```

**Linear algebra** (the foundation of neural networks):
```python
torch.matmul(x, y)        # Matrix multiply (works for any rank ≥ 2)
x @ y                     # Same as matmul (Python 3.5+)
torch.mm(x, y)            # 2D matrix multiply only
torch.bmm(x, y)           # Batched matrix multiply: (B, N, M) @ (B, M, P) → (B, N, P)
x.dot(y)                  # Dot product (1D only)
```

Understanding the different matrix multiplication functions is important: `mm()` only works for 2D matrices, `bmm()` is for batched 2D matrices, and `matmul()` is the general function that handles broadcasting and works for any dimensionality ≥ 1.

**Reductions** (collapse dimensions):
```python
x.sum(), x.mean(), x.std()
x.max(), x.min()
x.argmax(), x.argmin()     # Return indices, not values

# Reductions along specific dimensions
x.sum(dim=0)               # Sum along dimension 0 (collapse that dimension)
x.mean(dim=1, keepdim=True)  # Keep the reduced dimension as size 1
```

The `keepdim=True` option is crucial when you need the result to broadcast correctly with the original tensor.

### In-Place Operations: Modifying Memory Directly

Operations with a trailing underscore (`_`) modify the tensor in-place rather than creating a new tensor:

```python
x = torch.randn(3, 4)

x.add_(5)        # x += 5 (modifies x directly)
x.mul_(2)        # x *= 2
x.clamp_(0, 1)   # Clamp all values to [0, 1]
```

In-place operations save memory because they don't allocate new tensors, but they come with important caveats: you cannot use in-place operations on tensors that are leaves in the computation graph and require gradients, because doing so would break the chain of operations needed for backpropagation. When in doubt, use out-of-place operations (without the underscore); the performance difference is usually negligible, and you avoid subtle bugs.

---

## Device Management (CPU/GPU)

One of PyTorch's killer features is seamless GPU acceleration through a simple, unified API. However, with this power comes the responsibility of managing which device your tensors live on. Understanding device management is essential for leveraging GPU speed and avoiding common pitfalls.

### Moving Tensors Between Devices

PyTorch provides several equivalent ways to move tensors between CPU and GPU:

```python
x = torch.randn(3, 4)  # Created on CPU by default

# Move to GPU (these are equivalent)
x_gpu = x.to('cuda')
x_gpu = x.cuda()
x_gpu = x.to('cuda:0')  # Explicitly specify GPU 0

# Move back to CPU (these are equivalent)
x_cpu = x_gpu.to('cpu')
x_cpu = x_gpu.cpu()

# Move to specific GPU in multi-GPU system
x_gpu0 = x.to('cuda:0')
x_gpu1 = x.to('cuda:1')
```

The `.to()` method is more flexible because it can also change dtype, but `.cuda()` and `.cpu()` are more concise for simple device transfers. Importantly, these operations return a **new tensor on the target device**—they don't modify the original tensor in place (unless the tensor is already on that device, in which case `.to()` returns the same tensor).

### Best Practice: Device-Agnostic Code

Writing device-agnostic code means your code works whether or not a GPU is available, without modification. This is essential for portability and for developing on a laptop but training on a GPU server:

```python
# Set device based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create tensors directly on the target device
x = torch.randn(3, 4, device=device)

# Or create on CPU and move
y = torch.randn(3, 4).to(device)

# Your model
model = MyModel().to(device)

# In your training loop
for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    # ...
```

This pattern is so common that you should make it second nature. The key principle is to decide on a device once at the start of your script, then consistently use `.to(device)` for all tensors and models.

### GPU Memory Management

GPUs have limited memory (typically 8-48 GB depending on your hardware), and running out of GPU memory is one of the most common errors in deep learning:

```python
# Check current GPU memory usage
torch.cuda.memory_allocated()   # Bytes currently allocated to tensors
torch.cuda.memory_reserved()    # Bytes reserved by caching allocator

# Clear unused cached memory
torch.cuda.empty_cache()        # Frees unused cached memory back to GPU

# Context manager for specific GPU
with torch.cuda.device(0):
    x = torch.randn(1000, 1000, device='cuda')
```

Understanding the difference between allocated and reserved memory is important: PyTorch's caching allocator keeps memory reserved even after you delete tensors, so that future allocations can be fast. Calling `empty_cache()` releases this cached memory back to the GPU, which can help if you're running multiple processes or need to free up memory for another application, but it won't help if you're simply running out of memory because your model or batch size is too large.

### Performance Considerations

Moving data between CPU and GPU is **slow** compared to computation—it involves copying data across the PCIe bus, which has much lower bandwidth than either CPU or GPU memory. Some strategies to minimize this bottleneck:

1. **Batch transfers**: Move entire batches at once rather than individual examples
2. **Asynchronous transfers**: Use `x.to(device, non_blocking=True)` to allow CPU and GPU to work concurrently
3. **Keep data on GPU**: Once data is on the GPU, keep it there as long as possible
4. **Pin memory**: Use `pin_memory=True` in DataLoader for faster host-to-device transfers

```python
# Synchronous transfer (CPU waits for completion)
x_gpu = x.to('cuda')

# Asynchronous transfer (CPU can continue working)
x_gpu = x.to('cuda', non_blocking=True)

# This is particularly useful in data loading
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)
for data, target in dataloader:
    data = data.to(device, non_blocking=True)
    # GPU transfer happens in background while CPU prepares next batch
```

---

## Common Patterns and Idioms

Certain patterns appear over and over in PyTorch code. Recognizing these patterns will make you more efficient and help you understand others' code.

### Pattern 1: Batch Dimensions

Neural networks process data in batches for efficiency, which means adding an extra dimension to represent the batch:

```python
# Single sample: (channels, height, width)
image = torch.randn(3, 224, 224)

# Batch of images: (batch_size, channels, height, width)
batch = torch.randn(32, 3, 224, 224)

# Convert single sample to batch of size 1
image_batch = image.unsqueeze(0)  # (1, 3, 224, 224)

# Or using indexing syntax
image_batch = image[None, :, :, :]  # Same effect
```

This pattern is ubiquitous: models expect batched inputs, so even when you want to process a single example (like during inference), you need to add a batch dimension.

### Pattern 2: One-Hot Encoding

Converting class indices to one-hot vectors (where one element is 1 and all others are 0) is common for certain loss functions and visualizations:

```python
labels = torch.tensor([0, 2, 1])  # Class indices
num_classes = 3

one_hot = F.one_hot(labels, num_classes)
# tensor([[1, 0, 0],
#         [0, 0, 1],
#         [0, 1, 0]])
```

Many modern loss functions like `CrossEntropyLoss` expect class indices rather than one-hot vectors, but one-hot encoding is still useful for label smoothing and certain architectural patterns.

### Pattern 3: Masking and Conditional Operations

Applying different operations to different elements based on conditions:

```python
x = torch.randn(100)

# Create a boolean mask
mask = x > 0

# Extract elements satisfying condition
positive_values = x[mask]

# Conditional replacement (like ReLU: keep positive, zero negative)
x = torch.where(mask, x, torch.zeros_like(x))

# Or in-place
x[~mask] = 0  # Set all elements where mask is False to zero
```

This pattern is the vectorized, efficient alternative to Python loops with if-statements.

### Pattern 4: Concatenation vs Stacking

Combining multiple tensors along existing or new dimensions:

```python
x = torch.randn(3, 4)
y = torch.randn(3, 4)

# Concatenate along existing dimension (dimension must match except along concat dim)
torch.cat([x, y], dim=0)     # (6, 4) - stack vertically
torch.cat([x, y], dim=1)     # (3, 8) - stack horizontally

# Stack creates a NEW dimension
torch.stack([x, y], dim=0)   # (2, 3, 4) - new batch dimension
torch.stack([x, y], dim=1)   # (3, 2, 4)
```

The key difference: `cat()` requires tensors to have the same number of dimensions and concatenates along an existing dimension, while `stack()` creates a new dimension and requires tensors to have exactly the same shape.

---

## Common Pitfalls

Even experienced PyTorch users encounter these subtle bugs. Understanding them will save you hours of debugging.

### Pitfall 1: Unintended Broadcasting

Broadcasting can surprise you when dimensions don't match as you expect:

```python
# Intended: element-wise multiply of two column vectors
x = torch.randn(10, 1)  # Column vector
y = torch.randn(10)     # Row vector (lacks the explicit second dimension)

result = x * y  # Result is (10, 10), not (10, 1)!
```

**Why?** Broadcasting interprets `y` as shape `(10,)`, which right-aligns to `(1, 10)` when broadcasting with `(10, 1)`. Both dimensions then expand: `(10, 1)` and `(1, 10)` → `(10, 10)`.

**Solution**: Be explicit about dimensions:
```python
result = x * y.unsqueeze(1)  # (10, 1) * (10, 1) = (10, 1)
# Or reshape y
result = x * y.view(-1, 1)   # Same effect
```

### Pitfall 2: In-Place Operations on Leaf Tensors with Gradients

```python
x = torch.randn(3, requires_grad=True)
x += 1  # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation
```

**Why?** In-place operations modify the tensor's data, which would break the computation graph needed for backpropagation. PyTorch protects against this by raising an error.

**Solution**: Use out-of-place operations:
```python
x = x + 1  # Creates a new tensor, preserves computation graph
```

### Pitfall 3: View on Non-Contiguous Tensor

```python
x = torch.randn(3, 4).t()  # Transpose creates non-contiguous tensor
y = x.view(2, 6)           # RuntimeError: view size is not compatible with input tensor's size and stride
```

**Why?** `.view()` requires contiguous memory because it just reinterprets the existing flat memory array. A non-contiguous tensor can't be reinterpreted this way without actually moving data.

**Solutions**:
```python
# Option 1: Make contiguous first (creates copy)
y = x.contiguous().view(2, 6)

# Option 2: Use reshape (copies only if necessary)
y = x.reshape(2, 6)
```

### Pitfall 4: Device Mismatch

```python
x = torch.randn(3, 4, device='cuda')
y = torch.randn(3, 4)  # Default: CPU

z = x + y  # RuntimeError: Expected all tensors to be on the same device
```

**Why?** PyTorch can't perform operations between tensors on different devices—it would require implicit copying that might be expensive and surprising.

**Solution**: Ensure both tensors are on the same device:
```python
z = x + y.to(x.device)
# Or explicitly
z = x + y.to('cuda')
```

This is why device-agnostic code with a single `device` variable is so important.

---

## Performance Considerations

Writing correct PyTorch code is the first step; writing efficient PyTorch code requires understanding performance implications of different operations.

### 1. Vectorize Operations: Avoid Python Loops

Python loops are **slow** compared to vectorized tensor operations. PyTorch's operations are implemented in C++ and CUDA and are highly optimized:

❌ **Slow: Python loop**
```python
result = []
for i in range(len(x)):
    result.append(x[i] * 2)
result = torch.stack(result)
```

✅ **Fast: Vectorized**
```python
result = x * 2
```

The vectorized version can be **100-1000x faster** because it processes all elements in parallel and avoids Python interpreter overhead.

### 2. Understand Views vs Copies

Views are nearly free (just metadata updates), while copies involve actual memory allocation and data transfer:

```python
# Fast: view (no data copy)
y = x.view(2, -1)

# May copy: reshape (copies only if necessary)
y = x.reshape(2, -1)

# Always copies
y = x.clone()
```

Prefer views when possible, but remember they share memory (modifications affect both tensors).

### 3. In-Place Operations Can Save Memory

When you don't need to preserve the original tensor and it's safe (no gradient tracking), in-place operations save memory by avoiding allocation:

```python
# Creates new tensor (allocates memory)
x = x + 1

# Modifies in-place (no allocation)
x += 1  # or x.add_(1)
```

However, the performance gain is usually modest, and in-place operations have risks (breaking computation graph, unexpected shared memory), so only use them when you've measured a benefit.

### 4. GPU Transfers Are Expensive: Batch Them

Moving data between CPU and GPU is slow (limited by PCIe bandwidth, ~16 GB/s), while GPU computation is fast (TFLOPS). Minimize transfers:

❌ **Bad: Many small transfers**
```python
for i in range(1000):
    x_gpu = x[i].cuda()  # Transfer individual element
    result = model(x_gpu)
```

✅ **Good: Batch transfer**
```python
x_gpu = x.cuda()  # Transfer entire batch once
for i in range(1000):
    result = model(x_gpu[i])  # Indexing on GPU is fast
```

### 5. Use Asynchronous Transfers

By default, CPU-to-GPU transfers are synchronous (CPU waits for completion). Asynchronous transfers allow CPU and GPU to work concurrently:

```python
# Synchronous: CPU blocked
x_gpu = x.to('cuda')

# Asynchronous: CPU continues, transfer happens in background
x_gpu = x.to('cuda', non_blocking=True)
```

This is particularly effective when combined with pinned memory in data loaders, allowing data loading and GPU transfer to overlap.

### 6. Contiguous Memory Access Is Faster

Modern CPUs and GPUs have sophisticated caching systems that work best with sequential memory access. Contiguous tensors benefit from these optimizations:

```python
x = torch.randn(1000, 1000)

# Contiguous access: fast
sum_cols = x.sum(dim=0)

# Non-contiguous after transpose: slower
x_t = x.t()
sum_rows = x_t.sum(dim=0)  # Actually summing columns of original, non-sequential access
```

This is usually a micro-optimization—PyTorch's kernels are well-optimized—but it matters for very large tensors or performance-critical code.

---

## Summary

### Key Takeaways

| Concept | Essential Understanding |
|---------|------------------------|
| **Tensors** | Multidimensional arrays characterized by five properties: data, shape, dtype, device, and layout. They're the universal data structure in PyTorch, representing everything from images to model weights. |
| **Memory Model** | Tensors are stored as flat 1D arrays with strides that define navigation. Understanding contiguity and the view vs copy distinction is essential for both correctness and performance. |
| **Broadcasting** | Automatic shape expansion following three rules: right-align shapes, dimensions of size 1 stretch, and missing dimensions are treated as size 1. Broadcasting enables concise, memory-efficient operations between tensors of different shapes. |
| **Devices** | Tensors can live on CPU or GPU, and operations require all tensors to be on the same device. Device-agnostic code using `torch.device` is best practice for portable, flexible code. |
| **Performance** | Vectorize operations to avoid Python loops, prefer views over copies, batch GPU transfers, and use asynchronous transfers when possible. Understanding these principles is the difference between code that runs in seconds versus hours. |

### Mental Model: Layers of Abstraction

Think of PyTorch tensors as having three conceptual layers:

1. **Logical structure**: The multidimensional shape you think about (rows, columns, depth, etc.)
2. **Physical memory**: The flat 1D array of bytes in RAM or GPU memory
3. **Metadata**: Shape, strides, dtype, device that connect the logical to physical

Operations like `.view()` and `.transpose()` only modify the metadata (fast), while operations like `.clone()` and arithmetic create new physical memory (slower). Broadcasting operates at the logical level, avoiding physical replication.

### Quick Checklist: Before Operating on Tensors

Before performing operations, mentally check:
- ✓ **Shapes compatible?** Can these broadcast, or do I need to reshape?
- ✓ **Same device?** Are all tensors on the same CPU or GPU?
- ✓ **Contiguous for view?** If using `.view()`, is the tensor contiguous?
- ✓ **View or copy?** Does this operation share memory or create independent data?
- ✓ **Gradients needed?** If this requires gradients, am I avoiding problematic in-place operations?

### Debugging Strategy

When you encounter tensor-related errors:
1. Print shapes: `print(x.shape, y.shape)` - most errors are shape mismatches
2. Check device: `print(x.device, y.device)` - device mismatches are the second most common error
3. Check contiguity: `print(x.is_contiguous())` - relevant for view errors
4. Check dtypes: `print(x.dtype, y.dtype)` - mismatched dtypes can cause subtle bugs
5. Visualize with small examples: Create toy tensors with simple values to understand what's happening

---

**Next**: [02_autograd.md](02_autograd.md) - Understanding automatic differentiation and the computation graph that enables neural network training through backpropagation

**Related**:
- [Quick Reference](quick_reference.md) - Concise tensor operations cheat sheet for quick lookups
- [Appendix: Memory Deep Dive](appendix_memory_deep_dive.md) - Advanced memory management topics including storage objects, memory pinning, and optimization strategies
