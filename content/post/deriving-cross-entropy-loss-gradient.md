---
title: "Deriving cross-entropy loss gradient"
date: 2019-04-26
draft: false
series: "Machine Learning & Data Science concepts"
---

**Note**: This post assumes familiarity of vector notations and strong knowledge of (partial) derivatives which is covered in a Multivariable Calculus class.

In this blog post, I'm going to show how to derive the gradient (vector of partial derivatives) of the cost function for logistic regression (also known as [cross entropy loss](https://en.wikipedia.org/wiki/Cross_entropy)). This is an important part of backpropagation step/gradient descent algorithm in training logistic regression model (which I have explained in an earlier [blog post]({{< ref "/post/logistic-regression-explained.md" >}})) as in order to update the model parameters, we need to know the gradient of the cost function with respect to them.

The cost function of interest is of the following form:

$$
J(w, b) = \frac{-1}{m}\sum_{i=1}^{m}y^{(i)}log(a^{(i)}) + (1-y^{(i)})log(1-a^{(i)})
$$

where $m$ is the number of samples in training data set, $y^{(i)}$ is ground truth label for training sample $i$, $w$ (a 1-D vector of same dimension as input (column) vectors $x^{(i)}$'s) and $b$ (a scalar denoting bias) are model parameters that we're trying to find partial derivatives of the cost function above with respect to (w.r.t.), $a^{(i)}$ is the output of sigmoid function $\sigma$ (which is the activation function used in logistic regression model):

$$
\begin{aligned}
    \text{(1)} \quad a^{(i)} &= \sigma (z^{(i)}) = \frac{1}{1+e^{-z^{(i)}}} \newline
    \text{(2)} \quad z^{(i)} &= w^Tx^{(i)} + b
\end{aligned}
$$

It is important to remember that $y^{(i)}$'s and $x^{(i)}$'s are given and hence can be treated as constants. In other words, $w$ and $b$ are the only variables we need to worry about.

So the gradient/partial derivatives we're interested in computing are:

$$
\nabla J = \begin{bmatrix}
    \frac{\partial J}{\partial w} \newline
    \frac{\partial J}{\partial b}
    \end{bmatrix}
$$

Let's first start with computing the gradient for cross entropy loss for a single sample:

$$
\ell (w, b) = y*log(a) + (1-y)*log(1-a)
$$

*Note*: I drop the superscript (i)'s for simplicity

From calculus, we know we can find the derivative of a composite function by "chaining" derivatives of functions that it is composed of (aka [chain rule](https://en.wikipedia.org/wiki/Chain_rule))


For partial derivative of $\ell$ w.r.t. $w$, chain rule is applied as follows

$$
\text{(3)} \quad \frac{\partial \ell}{\partial w} = \frac{\partial \ell}{\partial a} * \frac{\partial a}{\partial z} * \frac{\partial z}{\partial w}
$$

where $\frac{\partial a}{\partial z}$ is derivative of equation (1) where $a$ is function of variable $z$, $\frac{\partial z}{\partial w}$ is (partial) derivative of equation (2) where $z$ is a linear function of two variables $w$ and $b$

Let's compute these partial derivatives

$$
\begin{aligned}
    \text{(4)} \quad \frac{\partial \ell}{\partial a} &= (y*log(a) + (1-y)*log(1-a))^{\prime} \newline
    &= \frac{y}{a} + \frac{-(1-y)}{1-a} \quad \text{(using derivative of log)} \newline
    &= \frac{y-a}{a(1-a)}
    \newline
    \newline
    \text{(5)} \quad \frac{\partial a}{\partial z} &= (\frac{1}{1+e^{-z}})^{\prime} \newline
    &= \frac{e^{-z}}{(1+e^{-z})^2} \quad \text{(using derivative of fraction and exponential)} \newline
    &= \frac{1}{1+e^{-z}} (\frac{1+e^{-z}-1}{1+e^{-z}}) \newline
    &= \frac{1}{1+e^{-z}} (1 - \frac{1}{1+e^{-z}}) \newline
    &= a(1-a)
    \newline
    \newline
    \text{(6)} \quad \frac{\partial z}{\partial w} &= (w^Tx+b)^{\prime} \newline
    &= x
\end{aligned}
$$

Putting (4), (5) and (6) together to compute (3)

$$
\begin{aligned}
\frac{\partial \ell}{\partial w} &= \frac{y-a}{a(1-a)} * a(1-a) * x \newline
&= (y-a)x
\end{aligned}
$$

Similarly for $\frac{\partial \ell}{\partial b} = y-a$ (since $\frac{\partial z}{\partial b}=1$)

Going back to our original overal cost function over all samples, we can now compute its gradient as follows

$$
\begin{aligned}
\frac{\partial J}{\partial w} &= \frac{-1}{m} \sum_{i=1}^{m} \frac{\partial \ell^{(i)}}{\partial w} \newline
&= \frac{-1}{m} \sum (y^{(i)} - a^{(i)})x^{(i)} \newline
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial J}{\partial b} &= \frac{-1}{m} \sum_{i=1}^{m} \frac{\partial \ell^{(i)}}{\partial b} \newline
&= \frac{-1}{m} \sum (y^{(i)} - a^{(i)}) \newline
\end{aligned}
$$

Here's how these partial derivatives (denoted by dw and db) look like in logistic regression model implementation in python (using numpy arrays) from scratch

```python
def propagate(W, b, X, Y):
  # Forward pass
  num_samples = X.shape[0]
  A = sigmoid(np.dot(X, W) + b)
  cost = -1/num_samples * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

  # Backward pass
  dw = 1/m * np.dot(X, (A-Y).T)
  db = 1/m * np.sum(A-Y)

  grads = {
    "dW": dW
    "db": db
  }
  return cost, grads
```


<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
