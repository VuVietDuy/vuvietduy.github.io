---
layout: post
title: "Semantics and Sentiment analysis"
author: vuvietduy
categories: [NLP, Tutorial]
featured: false
published: true
image:
toc: true
excerpt: ""
---

## Introduction

- Understand semantic word vectors with SpaCy and Python
- Understand sentiment analysis
- Leverage sentiment analysis for text classification

In order to use Spacy's embedded word vectors, we must download the larger spacy english models

Full details can be found at https://spacy.io/usage/models

At the command line download the medium or large spacy english models:

```
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

Now that you have the larger models that contain the word vertors, let's discuss how word vectors are created.

- Word2vec is a two-layer neural net that processes text.
- Its input is a text corpus and its output is a set of vectors: feature vectors for words in that corpus.
- The purpose and usefulness of Word2vec is to group the vectors of similar words together in vectorspace
- That is, it detects similarities mathematically
- Word2vec creates vectors that are distributed numerical representations of word features, features such as the context of individual words.
- It does so without human intervention

## Hand-on Semantics and Word Vectors with Spacy

Download en_core_web_sm

```python
!python -m spacy download en_core_web_sm
```

```python
# Import library
import spacy
nlp = spacy.load('en_core_web_lg')
```
