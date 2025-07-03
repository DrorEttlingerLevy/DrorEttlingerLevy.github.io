---
title: 'Batch Normalization in Deep Learning - Stabilization, Scaling, and Undoing'
date: 2025-06-30
permalink: /posts/2025/06/Batch_normalization/
image: /images/blog/30062025_1.png
preview: >
  Batch Normalization speeds up training and stabilizes deep networks by normalizing layer outputs, not just inputs. It can also be canceled by some mathematical tricks.
header:
  teaser: /blog/30062025_1.png
---

**Batch Normalization** helps neural networks train *faster* and *more stable* by normalizing the outputs of each layer — not just the inputs.

<div class="note">
  <strong>Note:</strong> Batch normalization is applied during training and adjusts outputs using learned parameters to maintain expressive power.
</div>

## ❓ Why Normalize?

‘Normalization’ means squeezing values into a range like `[0, 1]`, and ‘standardization’ makes the data have mean `0` and standard deviation `1`.

When features are in different scales, the model **struggles to learn** — gradients can explode or vanish.

<div class="info">
  <strong>Important:</strong> Batch norm doesn’t just apply to inputs. It’s applied across **each layer’s activations**, and it's done for every mini-batch.
</div>

So instead of normalizing once, we normalize *throughout the network*, layer-by-layer.

![Layer Normalization Example](/images/blog/30062025.png)

**Source:**  
[AssemblyAI – What it is and how to implement it](https://www.youtube.com/watch?v=yXOMHOpbon8&ab_channel=AssemblyAI)  
[CodeEmporium – EXPLAINED!](https://www.youtube.com/watch?v=DtEq44FTPM4&ab_channel=CodeEmporium)

---

## 🔍 Why Scale and Shift After Normalization?

Just normalizing forces the outputs to have `mean = 0` and `std = 1`, which can limit the network’s flexibility.

So instead, we **add two trainable parameters**, `γᵢ` (scale) and `βᵢ` (shift), so the network can learn how much to adjust the normalization.

<div class="note">
  <strong>Note:</strong> These two values allow the model to undo the normalization if needed — or learn a new better one for the task.
</div>

---

## 🔄 Undoing Batch Normalization?

Let’s say we want to go back to the original `xᵢ`, like batch norm didn’t happen.

The full expression is:

$$
\hat{h}_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i
$$

To cancel batch norm, we want:

$$
\hat{h}_i = x_i
$$

So:

$$
x_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i
$$

Multiply both sides:

$$
x_i \cdot \sqrt{\sigma^2 + \varepsilon} = \gamma_i(x_i - \mu) + \beta_i \cdot \sqrt{\sigma^2 + \varepsilon}
$$

Now match the coefficients:

$$
\gamma_i = \sqrt{\sigma^2 + \varepsilon}, \quad \beta_i = \mu
$$

Substitute:

$$
\hat{h}_i = x_i
$$

<div class="info">
  <strong>Key Insight:</strong> You can fully cancel Batch Norm by setting <code>γᵢ = √(σ² + ε)</code> and <code>βᵢ = μ</code>.
</div>

---

## 🧮 Batch Norm Computational Graph

Here’s how the operations flow inside a BN layer:

![Computational Graph](/images/blog/30062025_1.png)

First, we calculate `μ` and `σ²` for the current batch. Then normalize `xᵢ`, and apply the learned `γ` and `β`.

During inference, we don’t use the batch stats anymore — instead, we use **running averages** collected during training.

---

**In summary**, Batch Normalization makes training more efficient, smoother, and often gives better generalization.  
But what’s nice is that it’s still flexible — the network can learn to apply it fully, partially, or *not at all* depending on the task.