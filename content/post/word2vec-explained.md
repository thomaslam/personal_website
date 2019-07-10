---
title: "Word2vec Explained"
date: 2019-07-08
draft: true
series: "Machine Learning & Data Science concepts"
---

Word2vec is a neural network model that takes as input a large text corpus and produces a set of vectors corresponding to each unique word in the input corpus. 

The reason why we are interested in representing words as vectors using word2vec (or any other techniques/models for that matter) is this representation is useful for downstream NLP tasks such as sentiment classification, information retrieval, part-of-speech tagging, Q&A system, etc. Any ML model specifically tailored for these tasks would need as inputs numeric representations of words  

What is special about word2vec's representation of words is 

Compare this to one-hot representation of words where a vector representation of a word in the corpus is just a vector of 0s except for 1 at the index the word appears in in our vocab. For 

Limitations of word2vec

Does not address polysemy