FROM ermaker/keras

RUN conda install -y \
    jupyter \
    matplotlib \
    seaborn 

RUN pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0-cp27-none-linux_x86_64.whl

RUN conda install -y scikit-learn

RUN conda install -c conda-forge librosa

RUN conda install -c mutirri -y blessings=1.6

RUN conda install -c conda-forge tqdm=4.14.0

RUN pip install python-igraph

RUN conda install -y pillow=3.4.1
