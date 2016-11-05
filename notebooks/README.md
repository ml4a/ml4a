# Installation

Based on [official Tensorflow OS setup](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html).

## Ubuntu 16.04+

### Install Python3 for development

1. Install build utilities

```bash
sudo apt install build-essential python3-dev python3-virtualenv virtualenv 
```

2. Create a virtual environment

```bash
virtualenv --system-site-packages ~/virtualenv
```

### Install Tensorflow

Ubuntu/Linux 64-bit, CPU only, Python 3.5
```bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl
pip3 install --upgrade $TF_BINARY_URL
```

Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
> Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Install from sources" below.

```bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc2-cp35-cp35m-linux_x86_64.whl
pip3 install --upgrade $TF_BINARY_URL
```

### Install Jupyter and other libraries

```bash
pip install jupyter matplotlib keras
```

## OS X

### Install Python3 for development

1. Install [Homebrew](http://brew.sh/) 

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

2. Install Python 3

```bash
brew install python3
```

3. Create a virtual environment

```bash
virtualenv --system-site-packages ~/virtualenv
```

Activate the environment (using bash):
```bash
source ~/virtualenv/bin/activate
```

### Install Tensorflow

Install Tensorflow (prefer CPU as GPU enabled requires installing Cuda).

Mac OS X, CPU only, Python 3.4 or 3.5:
```bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc2-py3-none-any.whl
pip3 install --upgrade $TF_BINARY_URL
```

Mac OS X, GPU enabled, Python 3.4 or 3.5:
> Requires CUDA toolkit

```bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.11.0rc2-py3-none-any.whl
pip3 install --upgrade $TF_BINARY_URL
```

### Install Jupyter and other libraries

```bash
pip install jupyter matplotlib keras
```

## Using the notebooks later

To use TensorFlow later you will have to activate the Virtualenv environment again:
```bash
source ~/virtualenv/bin/activate
```
