from setuptools import setup

setup(
    name='ml4a',
    version='0.1',
    description='A toolkit for making art with machine learning, including an API for popular deep learning models, recipes for combining them, and a suite of educational examples',
    url='http://github.com/ml4a/ml4a-guides',
    author='Gene Kogan',
    author_email='gene@genekogan.com',
    license='LGPL 2.0',
    packages=['ml4a'],
    install_requires=[
        'tqdm',
        'ipython',
        'ipywidgets',
        'gdown',
        'matplotlib',
        'moviepy',
        'numpy',
        'Pillow',
        'opencv-python',
        'imutils',
        'scikit-image',
        'scikit-learn',
        'bs4',
        'noise',
        'lxml',
        'dlib',
        'face_recognition',
        'torch',
        'torchvision',
        'tensorflow-gpu==1.15.0'
    ],
    zip_safe=False
)

