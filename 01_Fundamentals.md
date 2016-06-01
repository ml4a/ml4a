# Welcome

The purpose of these guides is to go a bit deeper into the details behind common machine learning methods, assuming little math background, and teach you how to use popular machine learning Python packages. In particular, we'll focus on the Numpy, TensorFlow, and Keras libraries.

I'll assume you have some experience programming with Python - if not, check out these guides first: [Learn Python the Hard Way](http://learnpythonthehardway.org/book/). It will really help to illustrate the concepts introduced here.

Numpy underlies most Python machine learning packages and is great for performing quick sketches or working through calculations. TensorFlow is rapidly gaining popularity and seems poised to become machine learning's standard library, given its GPU support, performance, development support (Google), and distributed computation abilities. It can be quite low-level, which is great for experimenting with novel algorithms, but can be tedious when you want to use existing battle-tested methods. Keras provides some higher-level interfaces on top of TensorFlow and makes it easy to move quickly with established methods.

These guides will present the formal math for concepts alongside Python code examples since this often (for me at least) is a lot easier to develop an intuition for. Each guide is also available as an iPython notebook for your own experimentation.

The guides are not meant to exhaustively cover the field of machine learning (I don't know if that's possible) but I hope they will instill you with the confidence and knowledge to explore further on your own.

If you do want more details, you might enjoy my [artificial intelligence notes](http://frnsys.com/ai_notes).

---

# Modeling the world

You've probably seen various machine learning algorithms pop up - linear regression, SVMs, neural networks, random forests, etc. How are they all related? What do they have in common? What is machine learning for anyways?

First, let's consider the general, fundamental problem all machine learning is concerned with, leaving aside the algorithm name soup for now. The primary concern of machine learning is _modeling the world_.

We can model phenomena or systems - both natural and artificial, if you want to make that distinction - with mathematical functions. We see something out in the world and want to describe it in some way, we want to formalize how two or more things are related, and we can do that with a function. The problem is, for a given phenomenon, how do we figure out what function to use? There are infinitely many to choose from!

Before this gets too abstract, let's use an example to make things more concrete.

Say we have a bunch of data about the heights and weights of a species of deer. We want to understand how these two variables are related - in particular, given the weight of a deer, can we predict its height?

You might see where this is going. The data looks like a line, and lines in general are described by functions of the form $y = mx + b$.

Remember that lines vary depending on what the values of $m$ and $b$ are:

![Varying lines](assets/lines.svg)

Thus $m$ and $b$ uniquely define a function - thus they are called the _parameters_ of the function - and when it comes to machine learning, these parameters are what we ultimately want to learn. So when I say there are infinitely many functions to choose from, it is because $m$ and $b$ can pretty much take on any value. Machine learning techniques essentially search through these possible functions to find parameters that best fit the data you have. One way machine learning algorithms are differentiated is by how exactly they conduct this search (i.e. how they learn parameters).

In this case we've (reasonably) assumed the function takes the form $y = mx + b$, but conceivably you may have data that doesn't take the form of a line. Real world data is typically a lot more convoluted-looking. Maybe the true function has a $sin$ in it, for example.

This is where another main distinction between machine learning algorithms comes in - certain algorithms can model only certain forms of functions. _Linear regression_, for example, can only model linear functions, as indicated by its name. Neural networks, on the other hand, are _universal function approximators_, which mean they can (in theory) approximate _any_ function, no matter how exotic. This doesn't necessarily make them a better method, just better suited for certain circumstances (there are many other considerations when choosing an algorithm).

For now, let's return to the line function. Now that we've looked at the $m$ and $b$ variables, let's consider the input variable $x$. A function takes a numerical input; that is $x$ must be a number of some kind. That's pretty straightforward here since the deer weights are already numbers. But this is not always the case! What if we want to predict the sales price of a house (I need better examples). A house is not a number. We have to find a way to _represent_ it as a number (or as several numbers, i.e. a vector, which will be detailed in a moment), e.g. by its square footage. This challenge of representation is a major part of machine learning; the practice of building representations is known as _feature engineering_ since each variable (e.g. square footage or zip code) used for the representation is called a _feature_.

If you think about it, representation is a practice we regularly engage in. The word "house" is not a house any more than an image of a house is - there is no true "house" anyways, it is always a constellation of various physical and nonphysical components.

That's about it - broadly speaking, machine learning is basically a bunch of algorithms that learn you a function, which is to say they learn the parameters that uniquely define a function.

# Vectors

In the line example before I mentioned that we might have multiple numbers representing an input. For example, a house probably can't be solely represented by its square footage - perhaps we also want to consider how many bedrooms it has (I don't know anything about buying houses so forgive me if this example doesn't make sense). How do we group these numbers together?

That's what _vectors_ are for (they come up for many other reasons too, but we'll focus on representation for now). Vectors, along with matrices and other tensors (which will be explained a bit further down), could be considered the "primitives" of machine learning.

The Numpy library is best for dealing with vectors (and other tensors) in Python, so before we go anywhere further, let's import `numpy`:

```python
import numpy as np
```

You may have encountered vectors before in high school or college - to use Python terms, a vector is like a list of numbers. The mathematical notation is quite similar to Python code, e.g. `[5,4]`, but `numpy` has its own way of instantiating a vector:

```python
v = np.array([5, 4])
```

$$
v = \begin{bmatrix} 5 \\ 4 \end{bmatrix}
$$

Vectors are usually represented with lowercase variables.

Note that we never specified how _many_ numbers (also called _components_) a vector has - because it can have any amount. The amount of components a vector has is called its _dimensionality_. The example vector above has two dimensions. The vector `x = [8,1,3]` has three dimensions, and so on. Components are usually indicated by their index (usually using 1-indexing), e.g. in the previous vector, $x_1$ refers to the value $8$.

"Dimensions" in the context of vectors is just like the spatial dimensions you spend every day in. These dimensions define a __space__, so a two-dimensional vector, e.g. `[5,4]`, can describe a point in 2D space and a three-dimensional vector, e.g. `[8,1,3]`, can describe a point in 3D space. As mentioned before, there is no limit to the amount of dimensions a vector may have (technically, there must be one or more dimensions), so we could conceivably have space consisting of thousands or tens of thousands of dimensions. At that point we can't rely on the same human intuitions about space as we could when working with just two or three dimensions. In practice, most interesting applications of machine learning deal with many, many dimensions.

We can get a better sense of this by plotting a vector out. For instance, a 2D vector `[5,0]` would look like:

![A vector](assets/vector.svg)

So in a sense vectors can be thought of lines that "point" to the position they specify - here the vector is a line "pointing" to `[5,0]`. If the vector were 3D, e.g. `[8,1,3]`, then we would have to visualize it in 3D space, and so on.

So vectors are great - they allow us to form logical groupings of numbers. For instance, if we're talking about cities on a map we would want to group their latitude and longitude together. We'd represent Lagos with `[6.455027, 3.384082]` and Beijing separately with `[39.9042, 116.4074]`. If we have an inventory of books for sale, we could represent each book with its own vector consisting of its id, price, remaining stock.

To use vectors in functions, there are a few mathematical operations you need to know.

### Basic vector operations

Vectors can be added (and subtracted) easily:

```python
np.array([6, 2]) + np.array([-4, 4])
# >>> array([2, 6])
```

$$
\begin{bmatrix} 6 \\ 2 \end{bmatrix} + \begin{bmatrix} -4 \\ 4 \end{bmatrix} = \begin{bmatrix} 6 + -4 \\ 2 + 4 \end{bmatrix} = \begin{bmatrix} 2 \\ 6 \end{bmatrix}
$$

However, when it comes to vector multiplication there are many different kinds.

The simplest is _vector-scalar_ multiplication:

```python
3 * np.array([2, 1])
# >>> array([6, 3])
```

$$
3\begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 \times 2 \\ 3 \times 1
\end{bmatrix} = \begin{bmatrix} 6 \\ 3 \end{bmatrix}
$$

But when you multiply two vectors together you have a few options. I'll cover the two most important ones here.

The one you might have thought of is the _element-wise product_, also called the _pointwise product_, _component-wise product_, or the _Hadamard product_, typically notated with $\odot$. This just involves multiplying the corresponding elements of each vector together, resulting in another vector:

```python
np.array([6, 2]) * np.array([-4, 4])
# >>> array([-24, 8])
```

$$
\begin{bmatrix} 6 \\ 2 \end{bmatrix} \odot \begin{bmatrix} -4 \\ 4 \end{bmatrix} = \begin{bmatrix} 6 \times -4 \\ 2 \times 4 \end{bmatrix} = \begin{bmatrix} -24 \\ 8 \end{bmatrix}
$$

The other vector product, which you'll encounter a lot, is the _dot product_, also called _inner product_, usually notated with $\cdot$ (though when vectors are placed side-by-side this often implies dot multiplication). This involves multiplying corresponding elements of each vector and then summing the resulting vector's components (so this results in a scalar rather than another vector).

```python
np.dot(np.array([6, 2]), np.array([-4, 4]))
# >>> -16
```

$$
\begin{bmatrix} 6 \\ 2 \end{bmatrix} \cdot \begin{bmatrix} -4 \\ 4 \end{bmatrix} = (6 \times -4) + (2 \times 4) = -16
$$

The more general formulation is:

```python
# a slow pure-Python dot product
def dot(a, b):
    assert len(a) == len(b)
    return sum(a_i * b_i for a_i, b_i in zip(a,b))
```

$$
\begin{aligned}
\vec{a} \cdot \vec{b} &= \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix} \cdot \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix} = a_1b_1 + a_2b_2 + \dots + a_nb_n \\
&= \sum^n_{i=1} a_i b_i
\end{aligned}
$$

Note that the vectors in these operations must have the same dimensions!

Perhaps the most important vector operation mentioned here is the dot product. We'll return to the house example to see why. Let's say want to represent a house with three variables: square footage, number of bedrooms, and the number of bathrooms. For convenience we'll notate the variables $x_1, x_2, x_3$, respectively. We're working in three dimensions now so instead of learning a line we're learning a _hyperplane_ (if we were working with two dimensions we'd be learning a plane, "hyperplane" is the term for the equivalent of a plane in higher dimensions).

Aside from the different name, the function we're learning is essentially of the same form as before, just with more variables and thus more parameters. We'll notate each parameter as $\theta_i$ as is the convention (you may see $\beta_i$ used elsewhere), and for the intercept (what was the $b$ term in the original line), we'll add in a dummy variable $x_0 = 1$ as is the typical practice (thus $\theta_0$ is equivalent to $b$):

```python

# this is so clumsy in python;
# this will become more concise in a bit
def f(x0, x1, x2, x3, theta0, theta1, theta2, theta3):
    return theta0 * x0\
        + theta1 * x1\
        + theta2 * x2\
        + theta3 * x3
```

$$
y = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3
$$

This kind of looks like the dot product, doesn't it? In fact, we can re-write this entire function as a dot product. We define our feature vector $x = [x_0, x_1, x_2, x_3]$ and our parameter vector $\theta = [\theta_0, \theta_1, \theta_2, \theta_3]$, then re-write the function:

```python
def f(x, theta):
    return x.dot(theta)
```

$$
y = \theta x
$$

So that's how we incorporate multiple features in a representation.

There's a whole lot more to vectors than what's presented here, but this is the ground-level knowledge you should have of them. Other aspects of vectors will be explained as they come up.

## Learning

So machine learning algorithms learn parameters - how do they do it?

Here we're focusing on the most common kind of machine learning - _supervised_ learning. In supervised learning, the algorithm learns parameters from data which includes both the inputs and the true outputs. This data is called _training_ data.

Although they vary on specifics, there is a general approach that supervised machine learning algorithms use to learn parameters. The idea is that the algorithm takes an input example, inputs it into the current guess at the function (called the _hypothesis_, notate $h_{\theta}$), and then checks how wrong its output is against the true output. The algorithm then updates its hypothesis (that is, its guesses for the parameters), accordingly.

"How wrong" an algorithm can vary depending on the _loss function_ it is using. The loss function takes the algorithm's current guess for the output, $\hat y$, and the true output, $y$, and returns some value quantifying its wrongness. Certain loss functions are more appropriate for certain tasks, which we'll get into later.

We'll get into the specifies of how the algorithm determines what kind of update to perform (i.e. how much each parameter changes), but before we do that we should consider how we manage batches of training examples (i.e. multiple training vectors) simultaneously.


## Matrices

__Matrices__ are in a sense a "vector" of vectors. That is, where a vector can be thought of as a logical grouping of numbers, a matrix can be thought of as a logical grouping of vectors. So if a vector represents a book in our catalog (id, price, number in stock), a matrix could represent the entire catalog (each row refers to a book). Or if we want to represent a grayscale image, the matrix can represent the brightness values of the pixels in the image.

```python
A = np.array([
    [6, 8, 0],
    [8, 2, 7],
    [3, 3, 9],
    [3, 8, 6]
])
```

$$
\mathbf A =
\begin{bmatrix}
6 & 8 & 0 \\
8 & 2 & 7 \\
3 & 3 & 9 \\
3 & 8 & 6
\end{bmatrix}
$$

Matrices are usually represented with uppercase variables.

Note that the "vectors" in the matrix must have the same dimension. The matrix's dimensions are expressed in the form $m \times n$, meaning that there are $m$ rows and $n$ columns. So the example matrix has dimensions of $4 \times 3$. Numpy calls these dimensions a matrix's "shape".

We can access a particular element in a matrix by its indices. Say we want to refer to the element in the $i$th row and the $j$th column:

```python
A[i,j]
```

$$
A_{i,j}
$$

### Basic matrix operations

Like vectors, matrix addition and subtraction is straightforward (again, they must be of the same dimensions):

```python
B = np.array([
    [8, 3, 7],
    [2, 9, 6],
    [2, 5, 6],
    [5, 0, 6]
])

A + B
# >>> array([[14, 11,  7],
#            [10, 11, 13],
#            [ 5,  8, 15],
#            [ 8,  8, 12]])
```

$$
\begin{aligned}
\mathbf B &=
\begin{bmatrix}
8 & 3 & 7 \\
2 & 9 & 6 \\
2 & 5 & 6 \\
5 & 0 & 6
\end{bmatrix} \\
A + B &=
\begin{bmatrix}
8+6 & 3+8 & 7+0 \\
2+8 & 9+2 & 6+7 \\
2+3 & 5+3 & 6+9 \\
5+3 & 0+8 & 6+6
\end{bmatrix} \\
&=
\begin{bmatrix}
14 & 11 & 7 \\
10 & 11 & 13 \\
5 & 8 & 15 \\
8 & 8 & 12
\end{bmatrix} \\
\end{aligned}
$$

Matrices also have a few different multiplication operations, like vectors.

_Matrix-scalar multiplication_ is similar to vector-scalar multiplication - you just distribute the scalar, multiplying it with each element in the matrix.

_Matrix-vector products_ require that the vector has the same dimension as the matrix has columns, i.e. for an $m \times n$ matrix, the vector must be $n$-dimensional. The operation basically involves taking the dot product of each matrix row with the vector:

```python
# a slow pure-Python matrix-vector product,
# using our previous dot product implementation
def matrix_vector_product(M, v):
    return [dot(row, v) for row in M]

# or, with numpy:
np.matmul(M,v)
```

$$
\mathbf M v =
\begin{bmatrix}
M_{1} \cdot v \\
\vdots \\
M_{m} \cdot v \\
\end{bmatrix}
$$

We have a few options when it comes to multiplying matrices with matrices.

However, before we go any further we should talk about the _tranpose_ operation - this just involves switching the columns and rows of a matrix. The transpose of a matrix $A$ is notated $A^T$:

```python
A = np.array([
        [1,2,3],
        [4,5,6]
    ])

np.transpose(A)
# >>> array([
#       [1, 4],
#       [2, 5],
#       [3, 6]
#     ])
```

$$
\begin{aligned}
\mathbf A &=
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix} \\
\mathbf A^T &=
\begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}
\end{aligned}
$$


For matrix-matrix products, the matrix on the lefthand must have the same number of columns as the righthand's rows. To be more concrete, we'll represent a matrix-matrix product as $A B$ and we'll say that $A$ has $m \times n$ dimensions. For this operation to work, $B$ must have $n \times p$ dimensions. The resulting product will have $m \times p$ dimensions.

```python
# a slow pure-Python matrix Hadamard product
def matrix_matrix_product(A, B):
    _, a_cols = np.shape(A)
    b_rows, _ = np.shape(B)
    assert a_cols == b_rows

    result = []
    # tranpose B so we can iterate over its columns
    for col in np.tranpose(B):
        # using our previous implementation
        result.append(
            matrix_vector_product(A, col))
    return np.transpose(result)
```

$$
\mathbf AB =
\begin{bmatrix}
A B^T_1 \\
\vdots \\
A B^T_p
\end{bmatrix}^T
$$

Finally, like with vectors, we also have Hadamard (element-wise) products:

```python
# a slow pure-Python matrix-matrix product
def matrix_matrix_hadamard(A, B):
    result = []
    for a_row, b_row in zip(A, B):
        result.append(
            zip(a_i * b_i
                for a_i, b_i
                in zip(a_row, b_row))

# or, with numpy:
A * B
```

$$
\mathbf A \odot B =
\begin{bmatrix}
A_{1,1} B_{1,1} & \dots & A_{1,n} B_{1,n} \\
\vdots & \dots & \vdots \\
A_{m,1} B_{m,1} & \dots & A_{m,n} B_{m,n}
\end{bmatrix}
$$

Like vector Hadamard products, this requires that the two matrices share the same dimensions.

## Tensors

We've seen vectors, which is like a list of numbers, and matrices, which is like a list of a list of numbers. We can generalize this concept even further, for instance, with a list of a list of a list of numbers and so on. What all of these structures are called are _tensors_ (i.e. the "tensor" in "TensorFlow"). They are distinguished by their _rank_, which, if you're thinking in the "list of lists" way, refers to the number of nestings. So a vector has a rank of one (just a list of numbers) and a matrix has a rank of two (a list of a list of numbers).

Another way to think of rank is by number of indices necessary to access an element in the tensor. An element in a vector is accessed by one index, e.g. `v[i]`, so it is of rank one. An element in a matrix is accessed by two indices, e.g. `M[i,j]`, so it is of rank two.

Why is the concept of a tensor useful? Before we referred to vectors as a logical grouping of numbers and matrices as a logical grouping of vectors. What if we need a logical grouping of matrices? That's what 3rd-rank tensors are! A matrix can represent a grayscale image, but what about a color image with three color channels (red, green, blue)? With a 3rd-rank tensor, we could represent each channel as its own matrix and group them together.

## Learning continued

When the current hypothesis is wrong, how does the algorithm know how to adjust the parameters?

Let's take a step back and look at it another way. The loss function measures the wrongness of the hypothesis $h_{\theta}$ - another way of saying this is the loss function is a function of the parameters $\theta$. So we could notate it as $L(\theta)$.

The minimum of $L(\theta)$ is the point where the parameters guess $\theta$ is least wrong (at best, $L(\theta) = 0$, i.e. a perfect score, though this is not always good, as will be explained later); i.e. the best guess for the parameters.

So the algorithm learns the best-fitting function by minimizing its loss function. That is, we can frame this as an optimization problem.

There are many techniques to solve an optimization problem - sometimes they can be solved analytically (i.e. by moving around variables and isolating the one you want to solve for), but more often than not we must solve them numerically, i.e. by guessing a lot of different values - but not randomly!

The prevailing technique now is called _gradient descent_, and to understand how it works, we have to understand derivatives.

## Derivatives

Derivatives are everywhere in machine learning, so it's worthwhile become a bit familiar with them. I won't go into specifics on differentiation (how to calculate derivatives) because now we're spoiled with automatic differentiation, but it's still good to have a solid intuition about derivatives themselves.

A derivative expresses a rate of (instantaneous) change - they are always about how one variable quantity changes with respect to another variable quantity. That's basically all there is to it. For instance, velocity is a derivative which expresses how position changes with respect to time. Another interpretation, which is more relevant to machine learning, is that a derivative tells us how to change one variable to achieve a desired change in the other variable. Velocity, for instance, tells us how to change position by "changing" time.

To get a better understanding of _instantaneous_ change, consider a cyclist, cycling on a line. We have data about their position over time. We could calculate an average velocity over the data's entire time period, but we typically prefer to know the velocity at any given _moment_ (i.e. at any _instant_).

Let's get more concrete first. Let's say we have data for $n$ seconds, i.e. from $t_0$ to $t_n$ seconds, and the position at any given second $i$ is $p_i$. If we wanted to get the rate of change in position over the entire time interval, we'd just do:

```python
positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, # moving forward
             9, 9, 9, 9, 9, 9, 9, 9, 9, 9, # pausing
             9, 8, 7, 6, 5, 4, 3, 2, 1, 0] # moving backwards
t_0 = 0
t_n = 29
(p[t_n] - p[t_0])/t_n
# >>> 0.0
```

$$
v = \frac{p_n - p_0}{n}
$$

This kind of makes it look like the cyclist didn't move at all. It would probably be more useful to identify the velocity at a given second $t$. Thus we want to come up with some function $v(t)$ which gives us the velocity at some second $t$. We can apply the same approach we just used to get the velocity over the entire time interval, but we focus on a shorter time interval instead. To get the _instantaneous_ change at $t$ we just keep reducing the interval we look at until it is basically 0.

Derivatives have a special notation. A derivative of a function $f(x)$ with respect to a variable $x$ is notated:

$$
\frac{\delta f(x)}{\delta x}
$$

So if position is a function of time, e.g. $p = f(t)$, then velocity can be represented as $\frac{\delta p}{\delta t}$. To drive the point home, this derivative is also a function of time (derivatives are functions of what their "with respect to" variable is).

Since we are often computing derivatives of a function with respect to its input, a shorthand for the derivative of a function $f(x)$ with respect to $x$ can also be notated $f'(x)$.

### The Chain Rule

A very important property of derivatives is the _chain rule_ (there are other "chain rules" throughout mathematics, if we want to be specific, this is the "chain rule of derivatives"). The chain rule is important because it allows us to take complicated nested functions and more manageably differentiate them.

Let's look at an example to make this concrete:

```python
def g(x):
    return x**2

def h(x):
    return x**3

def f(x):
    return g(h(x))

# derivatives
def g_(x):
    return 2*x

def h_(x):
    return 3*(x**2)
```

$$
\begin{aligned}
g(x) &= x^2 \\
h(x) &= x^3 \\
f(x) &= g(h(x)) \\
g'(x) &= 2x \\
h'(x) &= 3x^2
\end{aligned}
$$

We're interested in understanding how $f(x)$ changes with respect to $x$, so we want to compute the derivative of $f(x)$. The chain rule allows us to individually differentiate the component functions of $f(x)$ and multiply those to get $f'(x)$:

```python
def f_(x):
    return g_(x) * h_(x)
```

$$
\frac{df}{dx} = \frac{dg}{dh} \frac{dh}{dx}
$$

This example is a bit contrived (there is a very easy way to differentiate this particular example that doesn't involve the chain rule) but if $g(x)$ and $h(x)$ were really nasty functions, the chain rule makes them quite a lot easier to deal with.

The chain rule can be applied to nested functions ad nauseaum! You can apply it to something crazy like $f(g(h(u(q(p(x))))))$. In fact, with deep neural networks, you are typically dealing with function compositions even more gnarly than this, so the chain rule is cornerstone there.


### Partial derivatives and gradients

The functions we've looked at so far just have a single input, but you can imagine many scenarios where you'd want to work with functions with some arbitrary number of inputs (i.e. a _multivariable_ function), like $f(x,y,z)$.

Here's where _partial deriatives_ come into play. Partial derivatives are just like regular derivatives except we use them for multivariable functions; it just means we only differentiate with respect to one variable at a time. So for $f(x,y,z)$, we'd have a partial derivative with respect to $x$, i.e. $\frac{\partial f}{\partial x}$ (note the slightly different notation), one with respect to $y$, i.e. $\frac{\partial f}{\partial y}$, and one with respect to $z$, i.e. $\frac{\partial f}{\partial z}$.

That's pretty simple! But it would be useful to group these partial derivatives together in some way. If we put these partial derivatives together in a vector, the resulting vector is the _gradient_ of $f$, notated $\nabla f$ (the symbol is called "nabla").


### Higher-order derivatives

We saw that velocity is the derivative of position because it describes how position changes over time. Acceleration similarly describes how _velocity_ changes over time, so we'd say that acceleration is the derivative of velocity. We can also say that acceleration is the _second-order_ derivative of position (that is, it is the derivative of its derivative).

This is the general idea behind higher-order derivatives.

## Gradient descent

Once you understand derivatives, gradient descent is really, really simple. The basic idea is that we use the derivative of the loss $L(\theta)$ with respect to $\theta$ and figure out which way the loss is decreasing, then "move" the parameter guess in that direction.

## Linear regression

Let's pull together everything that we've seen so far and walkthrough an example of linear regression.


