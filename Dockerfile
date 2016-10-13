FROM ermaker/keras

RUN conda install -y \
    jupyter \
    matplotlib \
    seaborn

RUN conda install -y scikit-learn

RUN conda install -c anaconda -y pillow=3.4.1

RUN conda install -c hcc -y librosa=0.3.1 

RUN conda install -c mutirri -y blessings=1.6