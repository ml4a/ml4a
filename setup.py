from setuptools import setup, find_packages

packages = ['ml4a', 'ml4a.dataset', 'ml4a.utils', 'ml4a.models', 'ml4a.canvas', 'ml4a.models.submodules']
submodules_root = 'ml4a.models.submodules'
submodules = {
    'BASNet': ['model', 'pytorch_iou', 'pytorch_ssim'],
    'deepdream': [],
    'glow': ['demo'],
    'idinvert_pytorch': ['boundaries', 'boundaries.stylegan_bedroom256', 'boundaries.stylegan_ffhq256', 'boundaries.stylegan_tower256', 'models', 'utils'],
    'neural_style': [],
    'PhotoSketch': ['data', 'models', 'options', 'scripts', 'util'],
    'semantic-segmentation-pytorch': ['config', 'data', 'mit_semseg', 'mit_semseg.config', 'mit_semseg.lib', 'mit_semseg.lib.nn', 'mit_semseg.lib.nn.modules', 'mit_semseg.lib.nn.modules.tests', 'mit_semseg.lib.nn.parallel', 'mit_semseg.lib.utils', 'mit_semseg.lib.utils.data'],
    'SPADE': ['data', 'datasets', 'models', 'models.networks', 'models.networks.sync_batchnorm', 'options', 'trainers', 'util'],
    'stylegan2': ['dnnlib', 'dnnlib.tflib', 'dnnlib.tflib.ops', 'dnnlib.submission', 'dnnlib.submission.internal', 'metrics', 'training'],
    'tacotron2': ['text', 'waveglow'],
    'Wav2Lip': ['evaluation', 'evaluation.scores_LSE', 'face_detection', 'face_detection.detection', 'face_detection.detection.sfd', 'models'],
    'White-box-Cartoonization': ['index_files', 'test_code', 'test_code.saved_models', 'train_code', 'train_code.selective_search']
} 
install_requires = [
    'tqdm',
    'ipython',
    'ipywidgets',
    'gdown',
    'matplotlib',
    'moviepy',
    'numpy',
    'Pillow',
    'dill', 
    'opencv-python',
    'imutils', 
    'localimport',
    'scikit-image', 
    'scikit-learn', 
    'librosa',
    'bs4', 
    'noise', 
    'lxml', 
    'dlib', 
    'face_recognition', 
    'torch', 
    'torchvision', 
    'tensorflow-gpu==1.15.0'
]

for submodule, subfolders in submodules.items():
    submodule_packages = ['{}.{}'.format(submodules_root, submodule)]
    submodule_packages.extend(['{}.{}.{}'.format(submodules_root, submodule, f) for f in subfolders])
    packages.extend(submodule_packages)

setup(
    name='ml4a',
    version='0.1',
    description='A toolkit for making art with machine learning, including an API for popular deep learning models, recipes for combining them, and a suite of educational examples',
    url='http://github.com/ml4a/ml4a-guides',
    author='Gene Kogan',
    author_email='gene@genekogan.com',
    license='LGPL 2.0',
    packages=packages, 
    install_requires=install_requires,
    zip_safe=False
)

