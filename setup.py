from setuptools import setup

setup(name='neuralarttools',
      version='0.1',
      description='A collection of various utilities mainly useful for pixel-optimization based neural arts, including masks and image-distortion functions, video processing, datasets, and convenience functions for Jupyter notebooks.',
      url='http://github.com/genekogan/neural-art-tools',
      author='Gene Kogan',
      author_email='gene@genekogan.com',
      license='LGPL 2.0',
      packages=['neuralarttools'],
      install_requires=[
          'Pillow',
          'numpy',
          'moviepy',
          'opencv-python',
          'imutils',
          'noise',
          'scikit-learn',
          'IPython'
      ],
      zip_safe=False)

