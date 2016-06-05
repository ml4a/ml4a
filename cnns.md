---
layout: guide
title: Convolutional Neural Networks: MNIST classification with Keras
---

(this is an annotated version of Keras's [example MNIST CNN code](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py))

The MNIST classification task is a classic machine learning benchmark. The data includes 70,000 handwritten grayscale digits, and the task is to identify them. The digits run from 0 to 9 so this is a multiclass classification problem. In particular, there are 10 possible classes.

The MNIST classification task is sort of like a "hello world" for computer vision, so a solution can be implemented quickly with an off-the-shelf machine learning library.

Since convolutional neural networks have thusfar proven to be the best at computer vision tasks, we'll use the Keras library to implement a convolutional neural net as our solution. Keras provides a well-designed and readable API on top of both Theano and TensorFlow fast backends, so we'll be done in a surprisingly short amount of steps!

Because MNIST is such a common task, the dataset is included with many machine learning libraries. With Keras, you can load the dataset with just a couple of lines:

```python
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

n_train, height, width = X_train.shape
n_test, _, _ = X_test.shape

n_train, n_test, height, width
```

We have 60,000 28x28 training grayscale images and 10,000 28x28 test grayscale images.

Some preprocessing steps are required to get the data into the proper form for the convolutional net.

```python
from keras.utils.np_utils import to_categorical

# we have to preprocess the data into the right form
X_train = X_train.reshape(n_train, 1, height, width).astype('float32')
X_test = X_test.reshape(n_test, 1, height, width).astype('float32')

# normalize from [0, 255] to [0, 1]
X_train /= 255
X_test /= 255

# numbers 0-9, so ten classes
n_classes = 10

y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)
```

Keras makes it very easy to define a neural network. We first instantiate a sequential Keras model, meaning the components of the model come one after the other - e.g. layer by layer.

```python
from keras.models import Sequential
model = Sequential()
```

The general architecture of a convolutional neural network is:

- convolution layers, followed by pooling layers
- fully-connected layers
- a final fully-connected softmax layer

We'll follow this same basic structure and interweave some other components, such as dropout, to improve performance.

To begin, we start with our convolution layers. We first need to specify some architecture hyperparemeters:

- _How many filters do we want for our convolution layers?_ Like most hyperparameters, this is chosen through a mix of intuition and tuning. A rough rule of thumb is: the more complex the task, the more filters. (Note that we don't need to have the same number of filters for each convolution layer, but we are doing so here for convenience.)
- _What size should our convolution filters be?_ We don't want filters to be too large or the resulting matrix might not be very meaningful. For instance, a useless filter size in this task would be a 28x28 filter since it covers the whole image. We also don't want filters to be too small for a similar reason, e.g. a 1x1 filter just returns each pixel.
- _What size should our pooling window be?_ Again, we don't want pooling windows to be too large or we'll be throwing away information. However, for larger images, a larger pooling window might be appropriate (same goes for convolution filters).

```python
# number of convolutional filters
n_filters = 32

# convolution filter size
# i.e. we will use a n_conv x n_conv filter
n_conv = 3

# pooling window size
# i.e. we will use a n_pool x n_pool pooling window
n_pool = 2
```

Now we can begin adding our convolution and pooling layers.

We're using only two convolutiona layers because this is a relatively simple task. Generally for more complex tasks you may want more convolution layers to extract higher and higher level features.

For our convolution activation functions we use ReLU, which is common and effective.

The particular pooling layer we're using is a max pooling layer, which can be thought of as a "feature detector".

```python
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D

model.add(Convolution2D(
        n_filters, n_conv, n_conv,

        # apply the filter to only full parts of the image
        # (i.e. do not "spill over" the border)
        # this is called a narrow convolution
        border_mode='valid',

        # we have a 28x28 single channel (grayscale) image
        # so the input shape should be (1, 28, 28)
        input_shape=(1, height, width)
))
model.add(Activation('relu'))

model.add(Convolution2D(n_filters, n_conv, n_conv))
model.add(Activation('relu'))

# then we apply pooling to summarize the features
# extracted thus far
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
```

Then we can add dropout and our dense and output (softmax) layers.

```python
from keras.layers import Dropout, Flatten, Dense

model.add(Dropout(0.25))

# flatten the data for the 1D layers
model.add(Flatten())

# Dense(n_outputs)
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# the softmax output layer gives us a probablity for each class
model.add(Dense(n_classes))
model.add(Activation('softmax'))
```

We tell Keras to compile the model using whatever backend we have configured (Theano or TensorFlow). At this stage we specify the loss function we want to optimize. Here we're using categorical cross-entropy, which is the standard loss function for multiclass classification.

We also specify the particular optimization method we want to use. Here we're using Adam, which adapts the learning rate based on how training is going and improves the training process.

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

Now that the model is defined and compiled, we can begin training and fit the model to our training data.

Here we're training for only 10 epochs. This is plenty for this task, but for more difficult tasks, more epochs may be necessary.

Training will take quite a while if you are running this on a CPU. Generally with neural networks, and especially with convolutional neural networks, you want to train on a GPU for much, much faster training times. Again, this dataset is relatively small so it won't take a terrible amount of time, but more than you might want to sit around and wait for.

```python
# how many examples to look at during each training iteration
batch_size = 128

# how many times to run through the full set of examples
n_epochs = 10

# the training may be slow depending on your computer
model.fit(X_train,
          y_train,
          batch_size=batch_size,
          nb_epoch=n_epochs,
          validation_data=(X_test, y_test))
```

Now we can evaluate the model on the test data to get a sense of how we did.

```python
# how'd we do?
loss, accuracy = model.evaluate(X_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)
```