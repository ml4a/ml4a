This is a very simple neural network example to demonstrate how they work.

We will just use `numpy` to be fully transparent about each step.

In practice, you'd want to use a library like [TensorFlow](https://www.tensorflow.org/) or a higher-level one like [Keras](http://keras.io/) to save yourself time and effort, but this approach should make the details clear.

First we'll import `numpy`:

```python
import numpy as np
```

With machine learning we are trying to find a hidden function that describes data that we have. Here we are going to cheat a little and define the function ourselves and then use that to generate data. Then we'll try to "reverse engineer" our data and see if we can recover our original function.

```python
def our_function(X):
    params = np.array([[2, -1, 5]])
    return np.dot(X, params.T)

X = np.array([
    [4,9,1],
    [2,5,6],
    [1,8,3]
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

That's one training iteration! In reality, you would do this many, many times - feedforward, backpropagate, update weights, then rinse and repeat. But otherwise, that's the basics of a neural network.
