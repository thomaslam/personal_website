---
title: "Word2vec Explained"
date: 2019-07-08
draft: true
series: "Machine Learning & Data Science concepts"
---

Word2vec is a neural network model that takes as input a large text corpus and produces a set of vectors corresponding to each unique word in the input corpus. 

The reason why we are interested in representing words as vectors using word2vec (or any other techniques/models for that matter) is this representation is useful for downstream NLP tasks such as sentiment classification, information retrieval, part-of-speech tagging, Q&A system, etc. Any ML model specifically tailored for these tasks would need as inputs numeric representations of words, hence the need for word2vec. 

What is special about word2vec's representation of words is it captures semantics better than its more straightforward alternative, **one-hot encoding**, which basically represents 

Limitations of word2vec
For plain word2vec (without negative sampling), softmax step is very expensive computationally
Does not address polysemy

## Implementation from scratch using numpy

Links
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model

http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling