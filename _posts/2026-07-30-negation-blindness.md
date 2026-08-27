---
title: "I Tried to Save $0.40 on API Tokens, and My Local AI Gaslit Me"
date: 2026-06-03
permalink: /posts/2026/06/negation-blindness/
image: /images/blog/olama.png
preview: >
  A local Ollama model completely reversed the meaning of a sentence,
  turning "an important file" into "not important at all." That mistake
  led me to a surprisingly interesting problem in LLMs: negation blindness.
header:
  teaser: /images/blog/olama.png
---

I tried to save $0.40 on OpenAI API tokens, and my local AI decided to completely gaslight me.

I used an Ollama model for an agent I'm building for my personal use. I thought it would be more convenient than connecting to the OpenAI API, and it would also save some token costs, even though my usage isn't very heavy.

One of the files was described as "an important file," and the model summarized it as: "This file is not important at all and negligible, can be ignored."

I was a bit shocked because that's quite a serious mistake. I knew Ollama models are significantly weaker, but I didn't realize they were weak to the point of completely reversing the meaning by 180 degrees.

It is not a simple hallucination. Looking a bit into the literature, I found this paper about semantic inversion by Kim et al. (2025), "Semantic Inversion, Identical Replies: Revisiting Negation Blindness in Large Language Models."
They tested pairs of questions (3,200 paired examples), for example: "Who set his guitar on fire?" and "Who did not set his guitar on fire?" This is how they could isolate failures caused specifically by semantic inversion instead of missing knowledge.

They showed that models sometimes fail to apply the logical transformation required by negation or polarity. And indeed, they found that larger models had much less negation blindness.

But maybe more importantly, they stated that accuracy alone is misleading. A model can score highly on QA benchmarks while still frequently reversing meaning under logical transformations.

I will say that their research paper studies explicit negation, while I used "un" as the negation, but still, it is the same failure point.
They call this negation blindness and proved that this can happen even in foundation models like ChatGPT and Claude 3.5.

It was a short deep dive into this surprising issue I ran into, and these are the moments when we get to learn interesting things we hadn't thought about before.
Have you also encountered mistakes like this or cases of negation blindness lately?