---
layout: post
title: "Introduction to Generative AI and LLMs"
author: vuvietduy
categories: [Blog, Tutorial]
featured: false
published: true
image:
toc: true
excerpt: "Introduction to Generative AI and LLMs | Generative AI for Beginners - Microsoft"
---

This lession will cover

- Introduction to our startup idea and mission
- Generative AI and how we landed on the current technology landscape
- Inner working of a large language model.
- Main capabilities and practical use cases of Large Language Models

## How do large language models work

- Tokenizer, text to numbers: Large Langue Models receive a text as input and generate a text as output. However, being statistical models, they work much better with numbers than text sequences. That's why every input to the model is processed by a tokenizer, before being used by the core model. A token is a chunk of text - consisting of a variable number of characters, so the tokenizer's main task is splitting the input into an array of tokens. Then, each token is mapped with a token index, which is the interger encoding of the original text chunk
- Predicting output tokens:
- Selection process, probability distribution
