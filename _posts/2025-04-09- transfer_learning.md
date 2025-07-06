---
title: 'Choosing a Neural Network for Transfer Learning on Underwater Audio Mel Spectrograms'
date: 2025-04-09
permalink: /posts/2025/04/transfer_learning/
image: /images/blog/09042025.png
preview: >
  Neural network for transfer learning on mel spectrograms of underwater audio, comparing ResNet50, YAMNet, and VGG11—each bringing different strengths to the table when handling image-like acoustic data.
header:
  teaser: /blog/09042025.png
---
I’ve been working with neural networks for transfer learning on audio signals. I’m converting raw underwater sounds into mel spectrograms using STFT (short-time Fourier transform), which turns the data into 2D images—so using image-based CNNs as feature extractors makes sense, and I’ve seen papers where this approach worked well.


To do this, I’m considering a few different networks. **ResNet50** is deep, reliable, and good at picking up spectrogram patterns. It’s my baseline, though it might be a bit heavy for my current PC setup, so I’m thinking about dropping the lower layers since they’re less relevant to my data. 

Then there’s **YAMNet**, which is based on MobileNet and trained directly on audio via AudioSet, giving it a solid advantage with acoustic features it’s also lightweight, so runtime will be easier to manage.

Lastly, **VGG11** older and not as efficient, but its simplicity is nice when you want clean architecture without too much going on, especially for more interpretable results.

Right now, I’ll probably start by testing all three on my spectrogram dataset, beginning with ResNet50 and seeing how it does. But before I can do any of that, I still need to extract mel spectrograms from all my 4TB of data one step at a time 😎

![My helpful screenshot](/images/blog/09042025.png)