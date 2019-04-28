---
title: "Things to Consider When Designing and Tuning Neural Networks"
date: 2019-04-27T19:17:06-04:00
draft: true
---

# Steps of building a neural network model

# Choice of activation functions

Most people use ReLU because it deals with vanishing gradient problem (input too large leading to zero gradients => slow training time)

For binary classification problems, sigmoid function is recommended as activation function for output layer. But not for hidden layers

# Weight initialization 
Randomly instead of zeros. Symmetry breaking problem

# Dropout layers

# Normalization

# Architectures besides Fully Connected
Convolutional Neural Networks, Recurrent Neural Networks