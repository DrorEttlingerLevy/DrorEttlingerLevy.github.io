---
title: "Optimized Aquaculture Feeding through Matched-Filter Audio Signal Processing and Machine Learning"
collection: publications
category: manuscripts
permalink: /publication/2026-01-08-optimized-aquaculture-feeding
excerpt: "A passive acoustic monitoring framework combining matched-filter signal processing and machine learning to continuously quantify feeding intensity in gilthead seabream."
date: 2026-01-08
venue: "Computers and Electronics in Agriculture"
image: /images/publications/article_title.png
header:
  teaser: /images/publications/article_title.png
paperurl: https://www.sciencedirect.com/science/article/pii/S0168169926000074
citation: "Ettlinger-Levy, D., Kendler, S., Meiri Ashkenazi, I., Tal, S., & Fishbain, B. (2026). Optimized aquaculture feeding through matched-filter audio signal processing and machine learning. Computers and Electronics in Agriculture, 243, 111412."
---

Our paper, **"Optimized Aquaculture Feeding through Matched-Filter Audio Signal Processing and Machine Learning,"** was published in *Computers and Electronics in Agriculture*.

We developed a passive acoustic monitoring approach for continuously quantifying feeding intensity in gilthead seabream (*Sparus aurata*). The method extracts a species-specific bite sound template and applies matched filtering and sliding-window aggregation to transform continuous underwater audio into an interpretable numerical feeding-intensity signal.

Machine-learning models, including XGBoost and Random Forest, were then used to validate the resulting feeding-intensity measure against environmental and biological variables.

The framework provides a lightweight and scalable alternative to high-dimensional acoustic classification approaches, with the goal of supporting real-time feed optimization and fish welfare monitoring in commercial aquaculture.

[Read the full paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168169926000074)

[View the code and analysis repository on GitHub](https://github.com/DrorEttlingerLevy/Article-COMPAG-1125)

## Methodology

![Methodology workflow](/images/publications/workflow.png)

The workflow combines underwater acoustic acquisition, signal preprocessing, extraction of a biological bite template, matched filtering, sliding-window aggregation, and machine-learning validation.

## Experimental Setup

![Experimental setup](/images/publications/setup.png)

Continuous acoustic recordings were collected from gilthead seabream tanks using underwater hydrophones.

## Bite Template

![Gilthead seabream bite template](/images/publications/template.png)

A characteristic bite sound was extracted and used as the reference template for matched filtering.

## Acoustic Preprocessing

![Spectrograms before and after preprocessing](/images/publications/spectograms.png)

Spectral gating and high-pass filtering were used to suppress background and low-frequency noise while preserving the acoustic signatures associated with feeding.

## Parameter Optimization

![AUC optimization results](/images/publications/auc.png)

Sliding-window parameters and matched-filter thresholds were optimized using ROC-AUC analysis to identify settings that reliably separated feeding activity from background noise.

