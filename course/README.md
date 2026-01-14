# PyTorch Fundamentals Guide

> A comprehensive text-based reference for understanding PyTorch's core concepts, complementing the hands-on notebooks.

---

## About This Guide

This guide provides **conceptual depth** and **reference material** for PyTorch fundamentals. Unlike the interactive notebooks which focus on learning-by-doing, these documents emphasize:

- **Why things work the way they do**
- **Mental models** for understanding PyTorch's design
- **Common patterns** and best practices
- **Pitfalls** and how to avoid them

### How to Use This Guide

- **First-time learners**: Read in order (01 → 04) alongside the notebooks
- **Review**: Jump to specific sections via the table of contents
- **Reference**: Use the [Quick Reference](quick_reference.md) for syntax lookups
- **Teaching**: Use as lecture notes or reading material

---

## Table of Contents

### Core Guides

1. **[Tensors: The Foundation](01_tensors.md)** (~4000 words)
   - What tensors are and why they exist
   - Memory model: strides, contiguity, views vs copies
   - Broadcasting rules and shape manipulation
   - Device management (CPU/GPU)
   - Common operations and patterns
   - Performance considerations

2. **[Automatic Differentiation](02_autograd.md)** (~4000 words)
   - Dynamic computation graphs
   - Forward and backward passes
   - Gradient flow and accumulation
   - Context managers: `no_grad()`, `inference_mode()`, `enable_grad()`
   - Custom autograd functions
   - Debugging gradients

3. **[Modules: Building Neural Networks](03_modules.md)** (~4000 words)
   - The `nn.Module` architecture
   - Parameters, buffers, and submodules
   - Forward hooks and backward hooks
   - Initialization strategies
   - Model serialization (save/load)
   - train() vs eval() modes

4. **[Training Loop Mechanics](04_training.md)** (~4000 words)
   - Anatomy of a training loop
   - Loss functions and their properties
   - Optimizers: SGD, Adam, and variants
   - Learning rate scheduling
   - Gradient clipping and numerical stability
   - Best practices and common mistakes

### Supplementary Materials

- **[Quick Reference](quick_reference.md)** - One-page cheat sheet
- **[Appendix: Memory Deep Dive](appendix_memory_deep_dive.md)** - Advanced memory topics
- **[Appendix: Debugging Guide](appendix_debugging_guide.md)** - Systematic debugging strategies

---

## Learning Path

### Recommended Order

```
Start
  ↓
01_tensors.md ← Read this first
  ↓
02_autograd.md ← Then understand gradients
  ↓
03_modules.md ← Learn to build models
  ↓
04_training.md ← Put it all together
  ↓
Quick Reference ← Keep this handy
```

### Parallel Learning

For the best experience, alternate between:
- **Reading**: Understand concepts (these guides)
- **Practicing**: Run code (notebooks 01-04)
- **Reviewing**: Reference syntax (quick_reference.md)

---

## Key Concepts by Guide

| Guide | Core Question | Key Concepts |
|-------|--------------|--------------|
| **Tensors** | How is data represented? | Shape, dtype, device, strides, broadcasting |
| **Autograd** | How are gradients computed? | Computation graph, backward pass, requires_grad |
| **Modules** | How are models structured? | Parameters, forward(), hooks, state_dict |
| **Training** | How do models learn? | Loss, optimizer, scheduler, training loop |

---

## Design Philosophy

These guides follow these principles:

1. **Concept-first**: Explain *why* before *how*
2. **Mental models**: Build intuition through analogies and visualizations
3. **Practical**: Focus on real-world usage patterns
4. **Self-contained**: Each guide can be read independently
5. **Cross-referenced**: Links between related concepts

---

## Contributing

Found an error or have suggestions? This is part of the PyTorch Deep Learning Course project.

---

## Next Steps

👉 Start with **[01_tensors.md](01_tensors.md)** to understand PyTorch's foundation

Or jump to:
- [02_autograd.md](02_autograd.md) - Automatic differentiation
- [03_modules.md](03_modules.md) - Building neural networks
- [04_training.md](04_training.md) - Training mechanics
- [quick_reference.md](quick_reference.md) - Quick syntax lookup
