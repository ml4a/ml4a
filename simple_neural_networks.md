---
layout: guide
title: Neural Networks
---

Neural networks are currently the most prominent machine learning method, and not without good reason - they are very powerful. Here I will do a fairly quick introduction to their basics. I won't go into their history and I won't go too far into the technical weeds - just enough to equip you for learning more on your own. Additional guides will go deeper into specific architectures and approaches. I will try to be very explicit about what parts are "up in the air" (i.e. modifiable) so you get a sense of where you can experiment with new neural networks.

If you do want more details, I highly recommend Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), Goodfellow, Bengio, and Courville's [Deep Learning](http://www.deeplearningbook.org/) book, Yoav Goldberg's "[A Primer on Neural Network Models for Natural Language Processing](http://arxiv.org/abs/1510.00726)", and of course [Gene Kogan's introduction](http://ml4a.github.io/ml4a/neural_networks/). [My notes](http://frnsys.com/ai_notes/machine_learning/neural_nets.html) on neural networks include a lot more details and additional resources as well.

For the other neural network guides I will mostly rely on the excellent [Keras](http://keras.io/) library, which makes it very easy to build neural networks and can take advantage of [Theano](http://deeplearning.net/software/theano/) or [TensorFlow](https://www.tensorflow.org/)'s optimizations and speed. However, to demonstrate the basics of neural networks, I'll use `numpy` so we can see exactly what's happening every step of the way.

## What is a neural network?

A neural network is pretty simple - it's just a _network_ of _neurons_! That is, it is some neurons - often called "units", the terminology I'll try to stick to - that are connected in some way. Each unit is a little function: it takes some input from the (incoming) units that are connected to it and passes some output to the (outgoing) units it's connected to.

Generally these units are organized into layers, and each layer connects to the next. The most basic architecture (i.e. how the units are connected) is a _dense_ layer, in which every input is connected to every unit (these are also called _affine_ or _fully-connected_ layers). The input to a unit is a tensor consisting of the outputs of each incoming unit or it may be the input you give to the network (if they are in the first layer).

It's a vector if each incoming unit outputs a scalar, it's a matrix if each incoming unit outputs a vector, and so on.

![A feedforward neural network, [source](http://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Architecture/feedforward.html)](/guides/assets/feedforward.jpg)

To keep things simple, we'll just consider the case where the input is a vector (i.e. each unit outputs a scalar).

Each unit has a vector of weights - these are the parameters that the network learns.

The inner operations of the basic unit is straightforward. We collapse the weight vector $w$ and input vector $v$ into a scalar by taking their dot product. Often a _bias_ term $b$ is added to this dot product; this bias is also learned. then we pass this dot product through an _activation function_ $f$, which also returns a scalar. Activation functions are typically nonlinear so that neural networks can learn nonlinear functions. I'll mention a few common activation functions in a bit, but for now let's see what a basic unit is doing:

```python
def unit(v, w, b):
    return activation_function(np.dot(input, weights) + b)
```

$$
\text{output} = f(vw + b)
$$

Note that the output units often don't have an activation function.

Much of neural network research is concerned with figuring out better architectures (that is, how exactly the units are connected) and what the internals of a unit are (e.g. what activation function to use, but they can get more sophisticated than this).

You've probably heard of "deep learning" and all that really refers to is a neural network with multiple hidden layers. Given enough hidden layers (and nonlinear activation functions), a neural network can approximate _any_ function (hence they are "universal function approximators"). That sounds great and all, but the deeper a neural network gets, the harder it is to train, and the lack of a good training method for deep networks is what held the field back for some time.

Nowadays neural networks are trained using a method called _backpropagation_, which is essentially the chain rule of derivatives. Remember that "training" in the context of supervised machine learning means figuring out how to update the parameters based on the learner's error on the training data, and we normally accomplish this by computing the derivative of the learner's hypothesis.

We basically do the same thing to train a neural network!

It will become clearer when we understand what the hypothesis of a neural network is. Let's consider a very simple neural network with just three layers, where the input and hidden layers have two units each (I'm going to be very explicit, which will look a little clumsy).

## A basic neural network with `numpy`

First we'll import `numpy`:

```python
import numpy as np
```

With machine learning we are trying to find a hidden function that describes data that we have. Here we are going to cheat a little and define the function ourselves and then use that to generate data. Then we'll try to "reverse engineer" our data and see if we can recover our original function.

```python
def our_function(X):
    params = np.array([[2., -1., 5.]])
    return np.dot(X, params.T)

X = np.array([
    [4.,9.,1.],
    [2.,5.,6.],
    [1.,8.,3.]
])

y = our_function(X)
print(y)
```

Now we are going to setup our simple neural network. It will have just one hidden layer with two units (which we will refer to as unit 1 and unit 2).

First we have to define the weights (i.e. parameters) of our network.

We have three inputs each going into two units, then one bias value for each unit, so we have eight parameters for the hidden layer.

Then we have the output of those two hidden layer units going to the output layer, which has only one unit - this gives us two more parameters, plus one bias value.

So in total, we have eleven parameters.

Let's set them to arbitrary values for now.

```python
# hidden layer weights
hidden_layer_weights = np.array([
    [0.5, 0.5, 0.5],    # unit 1
    [0.1, 0.1, 0.1]     # unit 2
])
hidden_layer_biases = np.array([1. ,1.])

# output layer weights
output_weights = np.array([[1., 1.]])
output_biases = np.array([1.])
```

We'll use $\tanh$ activations for our hidden units, so let's define that real quick:

```python
def activation(X):
    return np.tanh(X)
```

$\tanh$ activations are quite common, but you may also encounter sigmoid activations and, more recently, ReLU activations (which output 0 when $x \leq 0$ and output $x$ otherwise). These activation functions have different benefits; ReLUs in particular are robust against training difficulties that come when dealing with deeper networks.

To make things clearer later on, we'll also define the linear function that combines a unit's input with its weights:

```python
def linear(input, weights, biases):
    return np.dot(input, weights.T) + biases
```

Now we can do a forward pass with our inputs $X$ to see what the predicted outputs are.

### Forward pass

First, we'll pass the input through the hidden layer:

```python
hidden_linout = linear(X, hidden_layer_weights, hidden_layer_biases)
hidden_output = activation(hidden_linout)
print('hidden output')
print(hidden_output)
```

(We're keeping the unit's intermediary value, `hidden_linout` for use in backpropagation.)

Then we'll take the hidden layer's output and pass it through the output layer to get our predicted outputs:

```python
output_linouts = linear(hidden_output, output_weights, output_biases)
output_outputs = output_linouts # no activation function on output layer

predicted = output_outputs
print('predicted')
print(predicted)
```

Now let's compute the mean squared error of our predictions:

```python
mse = np.mean((y - predicted)**2)
print('mean squared error')
print(mse)
```

Now we can take this error and backpropagate it through the network. This will tell us how to update our weights.

### Backpropagation

Since backpropagation is essentially a chain of derivatives (that is used for gradient descent), we'll need the derivative of our activation function, so let's define that first:

```python
def activation_deriv(X):
    return 1 - np.tanh(X)**2
```

Then we want to set a learning rate - this is a value from 0 to 1 which affects how large we tweak our parameters by for each training iteration.

You don't want to set this to be too large or else training will never converge (your parameters might get really big and you'll start seeing a lot of `nan` values).

You don't want to set this to be too small either, otherwise training will be very slow. There are more sophisticated forms of gradient descent that deal with this, but those are beyond the scope of this guide.

```python
learning_rate = 0.001
```

First we'll propagate the error through the output layer (I won't go through the derivation of each step but they are straightforward to work out if you know a bit about derivatives):

```python
# derivative of mean squared error
error = y - predicted

# delta for the output layer (no activation on output layer)
delta_output = error

# output layer updates
output_weights_update = delta_output.T.dot(hidden_output)
output_biases_update = delta_output.sum(axis=0)
```

Then through the hidden layer:

```python
# push back the delta to the hidden layer
delta_hidden = delta_output * output_weights * activation_deriv(hidden_linout)

# hidden layer updates
hidden_weights_update = delta_hidden.T.dot(X)
hidden_biases_update = delta_hidden.sum(axis=0)
```

Then we can apply the updates:

```python
output_weights -= output_weights_update * learning_rate
output_biases -= output_biases_update * learning_rate

hidden_layer_weights -= hidden_weights_update * learning_rate
hidden_layer_biases -= hidden_biases_update * learning_rate
```

That's one training iteration! In reality, you would do this many, many times - feedforward, backpropagate, update weights, then rinse and repeat. That's the basics of a neural network - at least, the "vanilla" kind. There are other more sophisticated kinds (recurrent and convolutional neural networks are two of the most common) that are covered in other guides.
