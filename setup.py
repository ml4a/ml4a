from setuptools import setup, find_packages

setup(
    name='ml4a',
    version='0.1',
    description='A toolkit for making art with machine learning, including an API for popular deep learning models, recipes for combining them, and a suite of educational examples',
    url='http://github.com/ml4a/ml4a-guides',
    author='Gene Kogan',
    author_email='gene@genekogan.com',
    license='LGPL 2.0',
    packages=['ml4a', 'ml4a.dataset', 'ml4a.utils', 'ml4a.models', 'ml4a.canvas', 'ml4a.models.submodules', 'ml4a.models.submodules.BASNet', 'ml4a.models.submodules.BASNet.model', 'ml4a.models.submodules.BASNet.pytorch_iou', 'ml4a.models.submodules.BASNet.pytorch_ssim', 'ml4a.models.submodules.glow', 'ml4a.models.submodules.glow.demo', 'ml4a.models.submodules.neural-style-pt', 'ml4a.models.submodules.stylegan2', 'ml4a.models.submodules.stylegan2.dnnlib', 'ml4a.models.submodules.stylegan2.dnnlib.tflib', 'ml4a.models.submodules.stylegan2.dnnlib.tflib.ops', 'ml4a.models.submodules.stylegan2.dnnlib.submission', 'ml4a.models.submodules.stylegan2.dnnlib.submission.internal', 'ml4a.models.submodules.stylegan2.metrics', 'ml4a.models.submodules.stylegan2.training'], 
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

