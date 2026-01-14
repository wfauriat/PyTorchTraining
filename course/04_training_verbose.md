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

Training a neural network is the process of iteratively adjusting its parameters to minimize a loss function that measures how well the network performs on a given task. This process forms the core of deep learning, and understanding how to implement it correctly and efficiently is essential for anyone working with PyTorch. Unlike some high-level frameworks that hide the training loop behind a single `.fit()` call, PyTorch gives you complete, explicit control over every step of the training process, which provides both flexibility and clarity.

The training process can be understood as a cycle with four fundamental stages that repeat for every batch of data:

1. **Forward pass**: Propagate input data through the network to compute predictions
2. **Loss computation**: Measure how far the predictions are from the desired outputs using a loss function
3. **Backward pass**: Compute gradients of the loss with respect to all model parameters through backpropagation
4. **Parameter update**: Use those gradients to adjust the parameters in a direction that reduces the loss

This cycle repeats thousands or millions of times across multiple **epochs** (complete passes through the training dataset), gradually improving the network's predictions. The beauty of PyTorch's approach is that this process is transparent and controllable—you write the training loop yourself, which means you can inspect, modify, or extend every part of it.

---

## Anatomy of a Training Loop

### The Five Essential Steps: The Heart of Training

Every PyTorch training loop follows a consistent pattern that has become so standard it's almost ritualistic. These five steps happen for every single batch of data:

```python
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. Zero gradients from previous iteration
        optimizer.zero_grad()

        # 2. Forward pass: compute predictions
        output = model(data)

        # 3. Compute loss: measure prediction error
        loss = criterion(output, target)

        # 4. Backward pass: compute gradients
        loss.backward()

        # 5. Update parameters using gradients
        optimizer.step()
```

Let's understand why each of these steps is absolutely essential and what goes wrong if you skip or misorder them:

| Step | Purpose | What Happens If Skipped | Why It Must Be In This Order |
|------|---------|------------------------|-----------------------------|
| **`optimizer.zero_grad()`** | Clear accumulated gradients from previous iteration | Gradients accumulate across batches, producing wildly incorrect gradient estimates and causing divergence | Must be first—gradients must be cleared before computing new ones |
| **Forward pass** | Compute predictions from current model | No output to measure or improve | Must come second—need predictions to compute loss |
| **Loss computation** | Quantify how wrong predictions are | No signal indicating what direction to update parameters | Must come third—need both predictions and targets to compute error |
| **`loss.backward()`** | Compute gradients via automatic differentiation | No gradients → parameters can't be updated → no learning occurs | Must come fourth—need loss to backpropagate from |
| **`optimizer.step()`** | Update parameters using computed gradients | Parameters don't change → model doesn't improve → learning doesn't happen | Must be last—need gradients before using them to update parameters |

The ordering is critical. For example, if you accidentally call `optimizer.step()` before `loss.backward()`, you'd be trying to update parameters using gradients that don't exist yet (or worse, stale gradients from a previous iteration). If you forget `optimizer.zero_grad()`, gradients from multiple batches accumulate, which can be useful in some advanced scenarios (gradient accumulation) but is usually a bug that causes training instability.

### Complete Training Loop: Putting It All Together

Here's a complete, realistic training loop that incorporates all the essential components you'd use in practice:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup: Create and configure all components
model = MyModel().to(device)  # Move model to GPU if available
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop: Iterate over epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode (affects dropout, batchnorm)
    running_loss = 0.0

    # Iterate over batches in this epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to same device as model (CPU or GPU)
        data, target = data.to(device), target.to(device)

        # The five essential steps
        optimizer.zero_grad()           # 1. Clear old gradients
        output = model(data)            # 2. Forward pass
        loss = criterion(output, target)  # 3. Compute loss
        loss.backward()                 # 4. Compute gradients
        optimizer.step()                # 5. Update parameters

        # Accumulate loss for logging (detach from graph to avoid memory leak)
        running_loss += loss.item()

    # Log statistics after each epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}: Average Loss = {avg_loss:.4f}')
```

**Important details in this code:**

- **`.to(device)`**: Ensures both model and data are on the same device (CPU or GPU). Trying to compute with tensors on different devices causes runtime errors.

- **`model.train()`**: Puts the model in training mode, which affects layers like dropout (which should drop activations during training) and batch normalization (which should update running statistics during training). This is essential even though it's easy to forget.

- **`loss.item()`**: Extracts the scalar value from the loss tensor. If you accumulate `loss` directly without `.item()`, you'd accumulate the entire computation graph history, causing a massive memory leak over the course of an epoch.

- **`len(train_loader)`**: The number of batches in an epoch, used to compute average loss. This equals `ceil(dataset_size / batch_size)`.

---

## Loss Functions

The **loss function** (also called the **objective function** or **criterion**) is a mathematical function that takes your model's predictions and the true targets and outputs a single scalar value measuring how wrong the predictions are. The entire training process is about minimizing this loss. Choosing the right loss function for your task is crucial because it determines what your model learns to optimize.

### Classification Losses: Discrete Predictions

When your task involves predicting discrete categories (like classifying images into dog breeds or predicting the next word in a sentence), you need classification loss functions.

#### Cross-Entropy Loss: The Standard for Multi-Class Classification

**Cross-entropy loss** (also called **log loss**) is the standard choice for multi-class classification when each example belongs to exactly one class out of multiple possible classes. It measures the dissimilarity between the predicted probability distribution and the true distribution.

```python
criterion = nn.CrossEntropyLoss()

# Model outputs raw scores (logits), NOT probabilities
logits = model(x)  # Shape: (batch_size, num_classes)

# Targets are class indices (integers from 0 to num_classes-1)
targets = torch.tensor([2, 0, 4, 1])  # Batch of 4 examples

# Compute loss
loss = criterion(logits, targets)
```

**Why logits, not probabilities?** For numerical stability. Computing softmax and then log has precision issues, so PyTorch's `CrossEntropyLoss` combines them internally using the **log-sum-exp trick** for numerical stability.

**What's happening under the hood:**

```python
# CrossEntropyLoss is equivalent to:
log_probs = F.log_softmax(logits, dim=1)  # Convert logits to log-probabilities
loss = F.nll_loss(log_probs, targets)      # Negative log-likelihood loss

# Which mathematically is:
# loss = -mean(log(softmax(logits)[i, targets[i]]) for i in batch)
```

The loss is higher when the model assigns low probability to the correct class and lower when it assigns high probability to the correct class. At the extreme, if the model assigns probability 1 to the correct class, the loss is 0; if it assigns probability near 0, the loss approaches infinity.

**When to use CrossEntropyLoss:**
- Image classification (MNIST, CIFAR-10, ImageNet)
- Text classification (sentiment analysis, topic classification)
- Any task where each example has exactly one correct class

#### Binary Cross-Entropy: For Binary and Multi-Label Classification

When you have only two classes (binary classification) or when each example can belong to multiple classes simultaneously (multi-label classification), you use **binary cross-entropy**:

```python
criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE for stability

# Binary classification: Model outputs a single logit
logits = model(x)  # Shape: (batch_size, 1) or just (batch_size,)
targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32)  # 0 or 1
loss = criterion(logits, targets)

# Multi-label classification: Model outputs multiple logits
logits = model(x)  # Shape: (batch_size, num_labels)
targets = torch.tensor([
    [1, 0, 1, 0, 1],  # Example 1 has labels 0, 2, 4
    [0, 1, 0, 0, 0],  # Example 2 has only label 1
], dtype=torch.float32)
loss = criterion(logits, targets)
```

**Why BCEWithLogitsLoss instead of BCELoss?** Numerical stability again. `BCEWithLogitsLoss` combines the sigmoid activation with the binary cross-entropy computation in a numerically stable way, while `BCELoss` expects you to apply sigmoid manually, which can lead to numerical issues.

**The mathematical difference from CrossEntropyLoss:** CrossEntropyLoss uses softmax (outputs sum to 1, enforcing mutual exclusivity), while BCEWithLogitsLoss uses sigmoid (each output is independent). This makes BCE suitable for multi-label problems where an example can have any combination of labels.

**When to use BCEWithLogitsLoss:**
- Binary classification (spam detection, yes/no decisions)
- Multi-label classification (image tagging where one image can have multiple tags like "outdoor", "people", "daytime")
- Semantic segmentation (each pixel classified independently)

### Regression Losses: Continuous Predictions

When your task involves predicting continuous values rather than discrete categories, you need regression loss functions.

#### Mean Squared Error (MSE): The Standard for Regression

**Mean Squared Error** penalizes the squared difference between predictions and targets:

```python
criterion = nn.MSELoss()

predictions = model(x)  # Continuous values
targets = torch.tensor([1.5, 2.3, 0.8, 3.1])

loss = criterion(predictions, targets)
# Mathematically: loss = mean((predictions - targets)^2)
```

**Properties of MSE:**
- **Differentiable everywhere**: Smooth gradient for optimization
- **Penalizes outliers heavily**: Errors are squared, so large errors contribute disproportionately
- **Units**: If your targets are in meters, MSE is in meters²

**When to use MSE:**
- Predicting continuous values (house prices, temperature, stock prices)
- When you care about large errors more than small errors
- When your data doesn't have significant outliers

#### Mean Absolute Error (MAE): More Robust to Outliers

**Mean Absolute Error** (also called **L1 loss**) uses absolute values instead of squaring:

```python
criterion = nn.L1Loss()

loss = criterion(predictions, targets)
# Mathematically: loss = mean(|predictions - targets|)
```

**Properties of MAE:**
- **Less sensitive to outliers**: Linear penalty instead of quadratic
- **Same units as targets**: If targets are in meters, MAE is in meters
- **Not differentiable at zero**: Though not a practical problem for SGD

**When to use MAE:**
- When your dataset has outliers you don't want to dominate training
- When you care about typical error magnitude
- When you want a loss with interpretable units

#### Smooth L1 Loss (Huber Loss): Best of Both Worlds

**Smooth L1 Loss** combines the best properties of MSE and MAE—it's quadratic for small errors (like MSE) but linear for large errors (like MAE):

```python
criterion = nn.SmoothL1Loss()

# Behavior:
# |error| < 1: loss = 0.5 * error^2  (like MSE)
# |error| ≥ 1: loss = |error| - 0.5  (like MAE)
```

This is popular in object detection (e.g., Faster R-CNN) because bounding box coordinates can have large errors early in training, and you don't want those to dominate.

**When to use Smooth L1:**
- Object detection and bounding box regression
- When you expect both small and large errors during training
- When you want the stability of MAE for outliers but the faster convergence of MSE for small errors

### Custom Loss Functions: Combining or Creating New Losses

You're not limited to built-in losses. Custom losses are common in research and specialized applications:

```python
def custom_loss(predictions, targets):
    """Combine multiple loss components."""
    mse = F.mse_loss(predictions, targets)
    mae = F.l1_loss(predictions, targets)

    # Weighted combination
    total_loss = mse + 0.1 * mae

    return total_loss

# Use like any criterion
loss = custom_loss(model(x), y)
loss.backward()
```

**Common custom loss patterns:**
- **Multi-task losses**: Weighted sum of losses for different tasks
- **Regularization losses**: Add penalties like L1/L2 on weights
- **Perceptual losses**: Losses computed in feature space rather than output space
- **Adversarial losses**: For GANs and adversarial training

### Loss Reduction: Controlling Aggregation

By default, PyTorch losses compute the mean over the batch, but you can control this:

```python
# Default: mean reduction
criterion = nn.MSELoss(reduction='mean')
loss = criterion(pred, target)  # Scalar

# Sum reduction
criterion = nn.MSELoss(reduction='sum')
loss = criterion(pred, target)  # Sum of all element-wise losses

# No reduction: per-sample losses
criterion = nn.MSELoss(reduction='none')
losses = criterion(pred, target)  # Shape: (batch_size,)

# Now you can apply custom weighting
sample_weights = torch.tensor([1.0, 2.0, 1.0, 0.5])  # Weight samples differently
weighted_loss = (losses * sample_weights).mean()
```

The `reduction='none'` option is powerful for:
- **Importance sampling**: Weight difficult examples more
- **Curriculum learning**: Gradually introduce harder examples
- **Per-sample analysis**: Understand which examples are hardest

---

## Optimizers

**Optimizers** implement algorithms for updating model parameters using the gradients computed during backpropagation. The choice of optimizer and its hyperparameters (especially learning rate) has an enormous impact on training speed, stability, and final performance.

### SGD: Stochastic Gradient Descent

**Stochastic Gradient Descent** is the simplest and oldest optimizer. Despite its simplicity, when properly tuned with momentum, SGD often achieves excellent final performance, particularly on vision tasks:

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

**The basic update rule:**
```
θ_{t+1} = θ_t - lr * ∇θ
```

Where `θ` represents parameters, `lr` is learning rate, and `∇θ` is the gradient. This simply takes a step in the direction opposite to the gradient, scaled by the learning rate.

**SGD with momentum** is almost always better than vanilla SGD:

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Momentum update rule:**
```
v_{t+1} = momentum * v_t + ∇θ      (accumulate velocity)
θ_{t+1} = θ_t - lr * v_{t+1}       (update parameters)
```

Momentum accumulates a velocity vector that smooths out the gradient direction over time. This provides several benefits:
- **Accelerates convergence** in consistent directions (gradients pointing the same way across steps)
- **Dampens oscillations** in directions where gradients fluctuate
- **Helps escape shallow local minima** by maintaining momentum through flat regions

**Weight decay (L2 regularization):**

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

Weight decay adds a penalty proportional to the squared magnitude of parameters, which encourages smaller weights and helps prevent overfitting.

**When to use SGD:**
- Computer vision tasks (image classification, object detection)
- When you can afford careful learning rate tuning
- When you want best final performance and have computational budget
- When combined with momentum (always use momentum!)

**Typical learning rates:** 0.1 to 0.01 (much higher than adaptive methods)

### Adam: Adaptive Moment Estimation

**Adam** is the most popular optimizer in deep learning because it adapts the learning rate for each parameter individually based on estimates of first and second moments of the gradients. This makes it much more forgiving to poor learning rate choices:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

**What Adam does:**
- Maintains an **exponential moving average of gradients** (first moment, like momentum)
- Maintains an **exponential moving average of squared gradients** (second moment)
- Adapts learning rate per parameter using these estimates

**Advantages of Adam:**
- **Works well with default hyperparameters** (lr=0.001, betas=(0.9, 0.999))
- **Automatically adapts learning rates** per parameter
- **Handles sparse gradients well** (useful for embeddings, NLP)
- **Fast initial progress** (usually converges faster than SGD early in training)

**Disadvantages of Adam:**
- **Can generalize worse than SGD+momentum** on some vision tasks
- **May not converge as well** (can get stuck in poor local minima)
- **Can overfit on small datasets** (adaptive LR can be too aggressive)

**When to use Adam:**
- NLP and Transformers (almost universally used)
- Initial prototyping and experiments (forgiving default hyperparameters)
- When you don't have time to tune learning rate carefully
- When working with sparse data or embeddings

**Typical learning rates:** 0.001 to 0.0001 (much lower than SGD)

### AdamW: Adam with Decoupled Weight Decay

**AdamW** is Adam with a crucial fix: it implements weight decay correctly by decoupling it from the gradient-based update. This seemingly small change often provides better generalization:

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Why AdamW > Adam:** Standard Adam incorporates weight decay into the adaptive learning rate computation, which dilutes its effect. AdamW applies weight decay directly to parameters, preserving its intended regularization effect.

**When to use AdamW:**
- **Transformers and large language models** (the standard choice)
- **When regularization is important** (weight decay actually works as intended)
- **Default choice for most modern architectures** (has largely replaced Adam)

This is now the default recommendation over vanilla Adam for nearly all use cases.

### Other Optimizers: Specialized Choices

```python
# RMSprop: Good for RNNs, used in many RL applications
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Adagrad: Adaptive, but learning rate decays over time (can be too aggressive)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# RAdam: Rectified Adam (more stable warm-up phase)
optimizer = optim.RAdam(model.parameters(), lr=0.001)
```

**RMSprop** is similar to Adam but uses only the second moment (no momentum term). It's popular in reinforcement learning and for RNNs.

### Optimizer Comparison and Recommendations

| Optimizer | Best For | Typical Learning Rate | Pros | Cons |
|-----------|----------|----------------------|------|------|
| **SGD + momentum** | Computer vision, when you can tune LR | 0.1 - 0.01 | Best final performance, simple | Requires careful LR tuning |
| **Adam** | Quick prototyping | 0.001 - 0.0001 | Works with defaults, fast initial convergence | Can generalize worse |
| **AdamW** | Transformers, NLP, modern architectures | 0.001 - 0.0001 | Proper weight decay, widely used | Slightly slower than Adam |
| **RMSprop** | RNNs, reinforcement learning | 0.001 | Good for non-stationary objectives | Less popular in modern deep learning |

**General recommendations:**
- **Starting a new project:** AdamW with lr=0.001
- **Vision tasks (CNNs):** Try both SGD (lr=0.1, momentum=0.9) and AdamW (lr=0.001)
- **NLP/Transformers:** AdamW with lr=0.0001-0.001
- **Research experimentation:** Adam/AdamW for fast iteration

### Parameter Groups: Different Learning Rates for Different Parts

Sometimes you want to optimize different parts of your model with different learning rates, which is common in transfer learning:

```python
optimizer = optim.Adam([
    {'params': model.pretrained_backbone.parameters(), 'lr': 1e-4},
    {'params': model.new_classifier.parameters(), 'lr': 1e-3}
], lr=1e-3)  # Default LR (used if not specified in a group)
```

This applies a lower learning rate (1e-4) to the pretrained backbone (which has already learned good features) and a higher learning rate (1e-3) to the new classifier (which is randomly initialized and needs to learn from scratch).

**Common parameter group patterns:**
- **Transfer learning:** Lower LR for pretrained layers, higher for new layers
- **Discriminative fine-tuning:** Gradually increasing LR from early to late layers
- **Separate weight decay:** Different regularization for different parameter types

---

## Learning Rate Scheduling

The **learning rate** is often called the most important hyperparameter in deep learning. Learning rate **scheduling** means changing the learning rate during training according to a predefined or adaptive policy. The intuition is simple: use high learning rates early for fast progress, then lower them later for fine-tuning and stability.

### Why Schedule the Learning Rate?

**High learning rate early in training:**
- Takes large steps in parameter space
- Makes rapid initial progress toward good solutions
- Helps escape poor initializations quickly

**Low learning rate later in training:**
- Takes smaller, more careful steps
- Allows fine-tuning near local optima
- Improves stability (prevents oscillation around the minimum)

Without scheduling, you're forced to choose a fixed learning rate that compromises between these two goals. Scheduling lets you have both.

### Common Schedulers

#### StepLR: Periodic Learning Rate Decay

**StepLR** reduces the learning rate by a fixed factor every fixed number of epochs:

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train_one_epoch(model, train_loader, criterion, optimizer)
    scheduler.step()  # Update LR after each epoch

# Learning rate schedule:
# Epochs 0-29:   lr = initial_lr (e.g., 0.1)
# Epochs 30-59:  lr = 0.1 * 0.1 = 0.01
# Epochs 60-89:  lr = 0.01 * 0.1 = 0.001
# Epochs 90+:    lr = 0.001 * 0.1 = 0.0001
```

**Parameters:**
- `step_size`: Number of epochs between LR reductions
- `gamma`: Multiplicative factor (LR is multiplied by gamma)

**When to use:** Simple baseline schedule, works reasonably for many tasks.

#### CosineAnnealingLR: Smooth Cosine Decay

**CosineAnnealingLR** decays the learning rate following a cosine curve, providing smooth decay:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,      # Number of epochs for one cycle
    eta_min=1e-6    # Minimum learning rate
)
```

**The schedule:** Learning rate smoothly decreases from initial value to `eta_min` following a cosine curve over `T_max` epochs.

**Advantages:**
- **Smooth decay** (no sudden drops like StepLR)
- **Well-studied** (popular in research, often works well)
- **Simple** (only two hyperparameters)

**When to use:** Default choice for many modern architectures (ResNets, Transformers).

#### OneCycleLR: Cycle Up Then Down

**OneCycleLR** increases LR from a low value to a maximum, then decreases it back down, all within a single training run:

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs
)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        # ... training step ...
        optimizer.step()

        scheduler.step()  # IMPORTANT: Update every batch, not every epoch!
```

**The schedule:**
```
LR
 │     /\
 │    /  \
 │   /    \___
 └────────────── Training steps
```

First half: LR increases from low to `max_lr`
Second half: LR decreases from `max_lr` to very low

**Why this works:** The increasing phase provides exploration (large steps), while the decreasing phase provides exploitation (fine-tuning). This often trains faster than constant LR.

**Critical detail:** OneCycleLR must be updated **every batch**, not every epoch, because the schedule is defined over the total number of training steps.

**When to use:** When you want faster convergence, popular in fast.ai community.

#### ReduceLROnPlateau: Adaptive Based on Metrics

**ReduceLROnPlateau** monitors a metric (like validation loss) and reduces LR when the metric stops improving:

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # 'min' for loss, 'max' for accuracy
    factor=0.1,       # Multiply LR by this when reducing
    patience=10,      # Wait this many epochs before reducing
    threshold=0.0001  # Minimum improvement to count as progress
)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    scheduler.step(val_loss)  # Pass the metric to monitor
```

**How it works:** If the metric doesn't improve for `patience` consecutive epochs, multiply LR by `factor`.

**Advantages:**
- **Adaptive:** Responds to actual training dynamics
- **Robust:** Reduces LR only when plateau is detected
- **Doesn't require knowing training length in advance**

**Disadvantages:**
- **Requires validation set evaluation** (adds computational cost)
- **Can be too conservative** (might reduce LR too late)

**When to use:** When you're uncertain about the best schedule, or when training time is variable.

### Scheduler Usage Patterns

```python
# Pattern 1: Per-epoch scheduling (most schedulers)
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, criterion, optimizer)
    scheduler.step()  # Update LR once per epoch

# Pattern 2: Per-batch scheduling (OneCycleLR)
for epoch in range(num_epochs):
    for batch in train_loader:
        # ... training step ...
        scheduler.step()  # Update LR every batch

# Pattern 3: Metric-based scheduling (ReduceLROnPlateau)
for epoch in range(num_epochs):
    train_one_epoch(...)
    val_metric = validate(...)
    scheduler.step(val_metric)  # Pass validation metric
```

**Critical:** Don't call `scheduler.step()` at the wrong frequency or it won't work as intended!

---

## Gradient Clipping

**Gradient clipping** prevents **exploding gradients** by limiting the maximum magnitude of gradients before they're used to update parameters. This is crucial for training stability in certain architectures, particularly RNNs and very deep networks.

### Why Gradients Explode

In deep networks, gradients are computed via the chain rule, which multiplies many partial derivatives together during backpropagation. If these derivatives are consistently larger than 1, their product grows exponentially with depth, causing gradients to "explode" to enormous values. When the optimizer uses these exploding gradients, it takes huge parameter updates that destabilize training, often causing loss to spike to NaN.

**Symptoms of exploding gradients:**
- Loss suddenly becomes NaN
- Loss spikes up dramatically after decreasing normally
- Parameters become NaN or Inf
- Training is unstable (wildly varying loss across iterations)

### Clip by Norm: The Standard Approach

**Clipping by norm** scales down the entire gradient vector if its norm exceeds a threshold, preserving its direction:

```python
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()  # Compute gradients

# Clip gradients before optimizer step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

**What it does:** Computes the global norm of all gradients combined. If this norm exceeds `max_norm`, all gradients are scaled down proportionally so the norm equals `max_norm`. If the norm is below `max_norm`, gradients are left unchanged.

**Mathematically:**
```
global_norm = sqrt(sum(param.grad.norm(2)^2 for all params))
if global_norm > max_norm:
    for param in params:
        param.grad = param.grad * (max_norm / global_norm)
```

**Why this is better than clipping by value:** Preserves the relative magnitudes and directions of gradients across parameters, just scaling them all uniformly.

### Clip by Value: Less Common

**Clipping by value** independently clamps each gradient element to a range:

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
# Every gradient element is clamped to [-0.5, 0.5]
```

This is less commonly used because it doesn't preserve gradient directions and can distort the gradient vector arbitrarily.

### Choosing Clipping Thresholds

**Typical values:**

```python
# RNNs and LSTMs: prone to exploding gradients
clip_grad_norm_(model.parameters(), max_norm=1.0)

# Transformers: larger and more stable
clip_grad_norm_(model.parameters(), max_norm=5.0)

# Very deep networks (100+ layers)
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**How to choose:** Monitor gradient norms during training:

```python
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(f"Gradient norm: {total_norm:.4f}")
```

Watch the maximum gradient norms you observe. Set `max_norm` slightly above typical norms but low enough to clip rare spikes.

### When to Use Gradient Clipping

**Always use for:**
- RNNs and LSTMs (almost always need clipping)
- Very deep networks (100+ layers)
- Reinforcement learning (notoriously unstable gradients)

**Consider using for:**
- Transformers (helps stability)
- GANs (both generator and discriminator)
- Any time you observe loss spikes or NaN losses

**Usually don't need for:**
- Shallow networks (< 10 layers)
- Well-behaved architectures with residual connections
- Very small learning rates (naturally limits updates)

---

## Best Practices

### 1. Reproducibility: Making Experiments Repeatable

Deep learning experiments can be frustratingly non-reproducible due to randomness in initialization, data shuffling, and GPU operations. Setting seeds ensures your results are repeatable:

```python
def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)                    # PyTorch CPU random seed
    torch.cuda.manual_seed_all(seed)           # PyTorch GPU random seeds (all GPUs)
    torch.backends.cudnn.deterministic = True  # Force deterministic CUDA operations
    torch.backends.cudnn.benchmark = False     # Disable cuDNN autotuner (non-deterministic)
    np.random.seed(seed)                       # NumPy random seed
    random.seed(seed)                          # Python random seed

set_seed(42)  # Call at the very start of your script
```

**Important tradeoff:** Setting `cudnn.deterministic = True` and `cudnn.benchmark = False` ensures reproducibility but can slow down training (cuDNN's fastest algorithms are sometimes non-deterministic). For research, prefer reproducibility; for production, benchmark matters more.

### 2. Mixed Precision Training: Faster Training with Less Memory

**Mixed precision training** uses 16-bit floating point (float16) for most operations while keeping sensitive operations in 32-bit (float32). On modern GPUs with Tensor Cores (V100, A100, RTX series), this can provide 2-3x speedup and significantly reduce memory usage:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    # Forward pass in mixed precision
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()

    # Optimizer step with unscaling
    scaler.step(optimizer)
    scaler.update()
```

**How it works:**
1. **`autocast()`**: Automatically casts operations to float16 where safe, keeps others in float32
2. **`scaler.scale(loss)`**: Scales loss by a large factor before backward (prevents float16 underflow)
3. **`scaler.step()`**: Unscales gradients before optimizer uses them
4. **`scaler.update()`**: Adjusts the scaling factor dynamically

**Benefits:**
- **~2x faster** on Tensor Core GPUs
- **~50% less memory** (can train larger models or bigger batches)
- **Minimal code changes** (just wrap forward pass)

**When to use:** Almost always on modern GPUs (V100+). The only reasons not to use it are older GPUs without Tensor Cores or specific numerical stability issues (rare).

### 3. Gradient Accumulation: Simulating Larger Batches

When your GPU memory is limited but you want the benefits of a larger batch size, **gradient accumulation** simulates large batches by accumulating gradients over multiple small batches before updating parameters:

```python
accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps

optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    output = model(data)

    # Scale loss by accumulation steps so gradients average correctly
    loss = criterion(output, target) / accumulation_steps

    # Accumulate gradients (don't clear them yet)
    loss.backward()

    # Update parameters every accumulation_steps batches
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Why divide loss by `accumulation_steps`?** Gradients add up linearly. If you accumulate 4 batches without scaling, your gradients are 4x larger than they should be. Dividing the loss by 4 makes the accumulated gradients equivalent to a single batch that's 4x larger.

**When to use:**
- GPU memory limited (can't fit desired batch size)
- Very large models (like Transformers with billions of parameters)
- When larger batch size improves training (common in many tasks)

### 4. Learning Rate Warm-up: Gradual Increase at Start

**Warm-up** gradually increases the learning rate from a small value to the target value over the first few epochs or steps. This prevents unstable updates early in training when parameters are randomly initialized and gradients are noisy:

```python
class WarmupScheduler:
    """Linearly increase LR from 0 to base_lr over warmup_steps."""
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        # Linear warmup: lr = base_lr * (step / warmup_steps)
        lr = self.base_lr * min(self.step_num / self.warmup_steps, 1.0)

        # Update learning rate in optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Usage
optimizer = optim.Adam(model.parameters(), lr=0.001)
warmup = WarmupScheduler(optimizer, warmup_steps=1000, base_lr=0.001)

for batch in train_loader:
    warmup.step()  # Gradually increase LR
    # ... training step ...
```

**When to use:**
- Transformers and large models (almost always use warmup)
- Large batch sizes (warmup stabilizes training)
- Adam/AdamW optimizers (particularly benefit from warmup)

**Typical warmup durations:** 1-10% of total training steps, or ~1000-5000 steps for Transformers.

### 5. Early Stopping: Preventing Overfitting

**Early stopping** monitors validation performance and stops training when it stops improving, preventing overfitting:

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience          # How many epochs to wait
        self.min_delta = min_delta        # Minimum improvement to count
        self.counter = 0                  # Epochs without improvement
        self.best_loss = None             # Best loss seen so far
        self.early_stop = False           # Flag to signal stopping

    def __call__(self, val_loss):
        if self.best_loss is None:
            # First epoch
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # No improvement (or improvement < min_delta)
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement! Reset counter
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(max_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break
```

**When to use:**
- Small datasets (prone to overfitting)
- When you're unsure how long to train
- When computational budget is limited (stop when not improving)

---

## Common Patterns

### Pattern 1: Complete Train and Validation Loop

Here's a production-quality training and validation pattern:

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()  # Set to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Five essential steps
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    # Compute epoch statistics
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()  # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass only
            output = model(data)
            loss = criterion(output, target)

            # Track statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# Main training loop
best_val_acc = 0.0

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )

    # Validate
    val_loss, val_acc = validate(
        model, val_loader, criterion, device
    )

    # Log
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

    # Update learning rate
    scheduler.step()
```

### Pattern 2: Checkpointing for Resumable Training

Save complete training state periodically so you can resume if interrupted:

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path='checkpoint.pth'):
    """Save complete training state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path='checkpoint.pth'):
    """Load and resume from checkpoint."""
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']

    return start_epoch, loss


# Save checkpoint every N epochs
for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    # Save regular checkpoint
    if (epoch + 1) % 10 == 0:
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss,
                       path=f'checkpoint_epoch_{epoch+1}.pth')

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss,
                       path='best_model.pth')
```

---

## Summary

### The Five-Step Training Mantra

This should become automatic:

```python
optimizer.zero_grad()     # 1. Clear old gradients
output = model(input)     # 2. Forward pass (compute predictions)
loss = criterion(...)     # 3. Compute loss (measure error)
loss.backward()           # 4. Backward pass (compute gradients)
optimizer.step()          # 5. Update parameters (learn)
```

### Key Decisions and Recommendations

| Decision | Default Recommendation | Alternative | When to Use Alternative |
|----------|----------------------|-------------|------------------------|
| **Loss Function** | CrossEntropyLoss (classification)<br>MSELoss (regression) | BCEWithLogitsLoss<br>L1Loss | Multi-label classification<br>Outlier-robust regression |
| **Optimizer** | AdamW | SGD + momentum | Quick experiments, NLP<br>Vision tasks, best final performance |
| **Learning Rate** | 0.001 (Adam)<br>0.01-0.1 (SGD) | Use LR finder | Depends heavily on task |
| **LR Scheduler** | CosineAnnealingLR | ReduceLROnPlateau | Standard choice<br>Adaptive, when uncertain |
| **Gradient Clipping** | max_norm=1.0 (RNNs)<br>max_norm=5.0 (Transformers) | None | RNNs, deep networks<br>Shallow, stable networks |
| **Mixed Precision** | Yes (on modern GPUs) | No | Almost always use<br>Only if old hardware |

### Troubleshooting Guide

| Problem | Possible Causes | Solutions |
|---------|----------------|-----------|
| **Loss not decreasing** | LR too low, wrong loss function, gradient not flowing | Increase LR, verify loss matches task, check requires_grad |
| **Loss → NaN** | LR too high, numerical instability, exploding gradients | Lower LR, gradient clipping, check for log(0) or division by zero |
| **Slow convergence** | LR too low, poor initialization, data issues | Increase LR, try better initialization (He, Xavier), normalize data |
| **Training loss decreasing but val loss increasing** | Overfitting | Add regularization (dropout, weight decay), more data, early stopping |
| **Both train and val loss high** | Underfitting (model too simple) | Bigger model, train longer, reduce weight decay |
| **Loss oscillating wildly** | LR too high, batch size too small | Lower LR, increase batch size, add gradient clipping |
| **Different results each run** | Non-deterministic operations | Set all seeds, enable cudnn.deterministic |

### Typical Training Pipeline Checklist

When setting up training for a new task:

✓ **Load and preprocess data** (normalize, augment if needed)
✓ **Split into train/val/test** (typically 80/10/10 or 70/15/15)
✓ **Create DataLoaders** (shuffle training data, pin_memory for speed)
✓ **Initialize model** and move to device
✓ **Choose loss function** matching your task
✓ **Choose optimizer** (AdamW as default, SGD+momentum for vision)
✓ **Set learning rate** (0.001 for Adam, 0.01-0.1 for SGD)
✓ **Add LR scheduler** (CosineAnnealing or ReduceLROnPlateau)
✓ **Add gradient clipping** if needed (RNNs, Transformers)
✓ **Enable mixed precision** on modern GPUs
✓ **Implement checkpointing** (save best model, resume capability)
✓ **Add validation loop** (monitor generalization)
✓ **Set up logging** (loss, metrics, learning rate)
✓ **Consider early stopping** to prevent overfitting

### Performance Optimization Checklist

To make training faster:

✓ **Use GPU** (30-100x faster than CPU)
✓ **Enable mixed precision** (2-3x speedup on modern GPUs)
✓ **Increase batch size** (up to GPU memory limit)
✓ **Set num_workers in DataLoader** (typically 4-8 for multi-core CPUs)
✓ **Use pin_memory=True** in DataLoader (faster CPU→GPU transfer)
✓ **Profile to find bottlenecks** (is it data loading, forward pass, or backward pass?)
✓ **Use cudnn.benchmark=True** if input sizes are fixed (faster cuDNN)
✓ **Compile model with torch.compile()** (PyTorch 2.0+, often 20-50% speedup)

---

**Previous**: [03_modules.md](03_modules.md) - Building neural networks with nn.Module
**Next**: [Quick Reference](quick_reference.md) - One-page cheat sheet of all essential PyTorch patterns

**Related**:
- PyTorch optimization documentation for advanced techniques
- Papers: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
- Papers: "Decoupled Weight Decay Regularization" (AdamW, Loshchilov & Hutter, 2017)
