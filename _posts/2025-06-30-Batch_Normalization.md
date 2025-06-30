---
title: 'Batch Normalization - Deep Dive'
date: 2025-06-30
permalink: /posts/2025/06/Batch_normalization/
image: /images/blog/30062025_1.png
preview: >
  Batch Normalization speeds up training and stabilizes deep networks by normalizing layer outputs, not just inputs. It's can also be canceled by some mathematical tricks.
header:
  teaser: /blog/30062025_1.png
---

## Intro
Normalization is collapsing the input to be between 0 and 1, unlike standardization that makes the mean 0 and the variance 1.  
It is hard for the NN (and for other ML algorithms) to learn the weights of the input features if they are in other scales.  

Problem - unstable gradient problem

Solution - instead of only normalizing our inputs and then feeding the data into our NN, we normalize all the outputs of all the layers in our network.  
![My helpful screenshot](/images/blog/30062025.png)

from: [Batch normalization What it is and how to implement it](https://www.youtube.com/watch?v=yXOMHOpbon8&ab_channel=AssemblyAI)  

This decreases the importance of the initial weights and the learning rate is faster and thus the training will be faster (even though we do extra calculation for the normalization each batch).  

from: [Batch Normalization - EXPLAINED!](https://www.youtube.com/watch?v=DtEq44FTPM4&ab_channel=CodeEmporium)

## Q: Why do we scale and shift (equation 2) the activations after normalizing them? 

The normalization itself is no enough and we need to also scale and shift the activations because the NN learns how it is best to normalize the activations based on the task and the data and according to what it learns. Normalizing alone forces outputs to zero mean and unit variance, which can restrict learning. By using the parameters for scale and shift, the network can recover the original distribution if needed, or learn a new optimal one.

## Q: Since 𝛾𝑖 and 𝛽𝑖 are parameters of the model, they can change during training. What values can these parameters take which will UNDO the Batch Normalization operation?

* I used ChatGPT to understand all the math, and wrote it in my steps of understanding

Let combine the equations (1) and (2) together:
$$
\hat{h}_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i
$$

if we want to undo the batch normalization that means that  $\hat{h}_i = x_i$ like no normalization happened.

so:
$$
x_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i
$$
$$
x_i \sqrt{\sigma^2 + \varepsilon} = \gamma_i(x_i - \mu) + \beta_i \sqrt{\sigma^2 + \varepsilon}
$$

If we want this to be true for all $x_i$ we need the coefficients to match on both sides.

On the right side it is 
$$
\sqrt{\sigma^2 + \varepsilon} * x_i
$$

and on the left side it is only 
$$
\gamma_i * x_i
$$

So to match we need:
$$
\gamma_i = \sqrt{\sigma^2 + \varepsilon}
$$

Now we need the coefficient of the $ \beta $ to be = 0, so:

$$
\beta_i - \frac{\gamma_i \mu}{\sqrt{\sigma^2 + \varepsilon}} = 0 \quad \Rightarrow \quad \beta_i = \frac{\gamma_i \mu}{\sqrt{\sigma^2 + \varepsilon}}
$$

Now  $\gamma_i $
$$
\beta_i = \frac{\sqrt{\sigma^2 + \varepsilon} \cdot \mu}{\sqrt{\sigma^2 + \varepsilon}} = \mu
$$

Finally:
Substituting back:
$$
\hat{h}_i = \sqrt{\sigma^2 + \varepsilon} \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \mu = x_i
$$

To summarize: To undo batch normalization, set $\gamma_i = \sqrt{\sigma^2 + \varepsilon}$ and $\beta_i = \mu$, which makes $\hat{h}_i = x_i$ ,matching coefficients on both sides of the equation so that no normalization effectively occurs.

## Q: Write a computational graph for the Batch Normalization layer.

![My helpful screenshot](/images/blog/30062025_1.png)
