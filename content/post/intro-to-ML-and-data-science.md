---
title: "Intro to ML and Data Science"
date: 2019-04-15T14:49:55-04:00
draft: true
series: "Machine Learning & Data Science concepts"
---

In this introductory post, I introduce what Machine Learning is in hopefully a widely accessible language. I aim to minimize as many buzzwords as possible though certain terminologies such as matrices, vectors, statistical models, parameters, etc. will come up. A basic understanding of what these terms mean is assumed.

# What is Machine Learning?
According to computer scientist Arthur Samuel in 1959, Machine Learning is a subfield of Computer Science that gives “computers the ability to learn without being explicitly programmed.” 

Let's dissect this statement. Normally a computer program that you interact with everyday (e.g. web browser, Word Document editing program, Excel, iPhone apps, etc.) is written by a coder to respond to some input using arbitrary rules and logic defined by the said coder and perform whatever its indended functionalities are. Think: when a user taps on a button on an app, the coder must write logic for the app to respond to such input.

What about tasks that don't have such straightforward logic such as classifying whether an image contains cats or not? For humans, this kind of task comes naturally but not so for computers. After all, an image is basically a matrix of 0s and 1s to a computer. Put yourself into a computer's shoes, how can you recognize an arbitrary such matrix contains a cat in it or not? More generally, how can we write computer programs that achieve human-level performance in this kind of task?

![How a computer views an image](/images/classify_cats.png)

The answer, at least for now, lies in writing computer programs that make decisions relying on outputs from statistical models which extract insights/patterns from large quantities of data available. _**Machine Learning is the scientific study of these statistical models.**_

Going back to classifying cat images task, instead of manually defining rules (which is impossible anyway) about what a cat image should look like, you can build a Machine Learning (ML) model that has been trained on a . you can collect all labelled cat and non-cat images from the and form a matrix of training data, where each row is a feature vector representing an image. You, as a programmer, only need to input this training data into your Machine Learning model, from which it can automatically learn model parameters using some training algorithm. Then for any unforeseen image (i.e. not in training data) fed into your model, based on model parameters learned during training, your model can predict whether the given image contains cats or not. Your computer program, utilizing such cat-classifying ML model which is trained on the prepared cat images data set, outputs the desired answer based on outputs of the model.

# Major categories of ML
There is a wide variety of tasks that ML algorithms/models can solve. The cat image classifying task above is an example of one category of ML tasks called *Supervised Learning*

# ML hierarchy within AI
Machine Learning is a subset of Artificial Intelligence, which has a more general definition: _**AI is the scientific study and development of computer systems that can perform tasks that normally require human intelligence**_. Machine Learning is only one way of making computers able to achieve human-level performance on such tasks by relying on statistical models. 

Here's a 

# ML's relationship with Data Science

# How is ML/DS useful?

A simple search... Here's a non-exhaustive list:

* Spam email filtering
* Movie/Ads/Product recommendation
* Chatbots/Question and Answering systems
* Online fraud detection
* Virtual assistants (e.g. Siri, Alexa, Google Now)
* Medical Diagnosis

There's hardly 


# Resources
More elegant introductions to this topic have been around on the internet. Here are a few of those resources which this post owes much of its content from:

* [Andrew Ng's Coursera course on Machine Learning](https://www.coursera.org/learn/machine-learning)
* [machinelearningmastery.com's intro post to ML](https://machinelearningmastery.com/what-is-machine-learning/)