---
layout: guide
title: Neural Networks
---

Neural networks are currently the most prominent machine learning method, and not without good reason - they are very powerful. Here I will do a fairly quick introduction to their basics. I won't go into their history and I won't go too far into the technical weeds - just enough to equip you for learning more on your own. Additional guides will go deeper into specific architectures and approaches. I will try to be very explicit about what parts are "up in the air" (i.e. modifiable) so you get a sense of where you can experiment with new neural networks.

If you do want more details, I highly recommend Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), Goodfellow, Bengio, and Courville's [Deep Learning](http://www.deeplearningbook.org/) book, Yoav Goldberg's "[A Primer on Neural Network Models for Natural Language Processing](http://arxiv.org/abs/1510.00726)", and of course Gene Kogan's introduction (TODO link!). [My notes](http://frnsys.com/ai_notes/machine_learning/neural_nets.html) on neural networks include a lot more details and additional resources as well.

For these neural network guides I will mostly rely on the excellent Keras library, which makes it very easy to build neural networks and can take advantage of Theano or TensorFlow's optimizations and speed. To illustrate some lower-level parts, I'll use Numpy.

## What is a neural network?

A neural network is pretty simple - it's just a _network_ of _neurons_! That is, it is some neurons - often called "units", the terminology I'll try to stick to - that are connected in some way. Each unit is a little function: it takes some input from the (incoming) units that are connected to it and passes some output to the (outgoing) units it's connected to.

Generally these units are organized into layers, and each layer connects to the next. The most basic architecture (i.e. how the units are connected) is a _dense_ layer, in which every input is connected to every unit (these are also called _affine_ or _fully-connected_ layers). The input to a unit is a tensor consisting of the outputs of each incoming unit or it may be the input you give to the network (if they are in the first layer).

It's a vector if each incoming unit outputs a scalar, it's a matrix if each incoming unit outputs a vector, and so on.

# TODO feed forward neural network image

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

It will become clearer when we understand what the hypothesis of a neural network is. Let's consider a very simple neural network with just three layers, where the input and hidden layers have two units each (I'm going to be very explicit, which will look a little clumsy). I'll ignore the biases because they don't change much.

The input layer will look like this:

```python

weights = [
    [0, 0],
    [0, 0],
    [0],
]

activation_function = ##

def input_layer(input):
    unit1_output = activation_function(np.dot(input, weights[0][0]))
    unit2_output = activation_function(np.dot(input, weights[0][1]))
    return [unit1_output, unit2_output]
```

$$
o_1 = [f(x w_{1,1}), f(x w_{1,2})]
$$

The hidden layer takes the first layer's outputs as input:

```python
def hidden_layer(layer1_output):
    unit1_output = activation_function(np.dot(layer1_output, weights[1][0]))
    unit2_output = activation_function(np.dot(layer1_output, weights[1][1]))
    return [unit1_output, unit2_output]
```

$$
o_2 = [f(o_1 w_{2,1}), f(o_1 w_{2,2})]
$$

The output layer takes the hidden layer's outputs as input:

```python
def output_layer(layer2_output):
    return np.dot(layer2_output, weights[2][0])
```

$$
o_3 = f(o_2 w_{3,1})
$$

So we could rewrite the network as simply:

```python
def network(input):
    return output_layer(hidden_layer(input_layer(input)))
```

# TODO clarify this
$$
k(h(g(x,w),w),w)
$$

This nesting of functions is the hypothesis that the neural network learns, and like with other learners, we can compute the derivative of the hypothesis with respect to the parameters. The chain rule of derivatives makes this straightforward:

# TODO direct example
$$
\frac{df}{dx} = \frac{dg}{dh} \frac{dh}{dx}
$$
