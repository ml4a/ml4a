import pathlib
from setuptools import setup, find_packages

packages = ['ml4a', 'ml4a.dataset', 'ml4a.utils', 'ml4a.models', 'ml4a.canvas', 'ml4a.models.submodules']
submodules_root = 'ml4a.models.submodules'

submodules = {
    'BASNet': ['model', 'pytorch_iou', 'pytorch_ssim'],
    'deepdream': [],
    'ESRGAN': ['models'],
    'face-parsing-PyTorch': ['modules', 'modules.src', 'modules.src.utils'],     
    'FlowNetPytorch': ['datasets', 'models'],
    'glow': ['demo'],
    'idinvert_pytorch': ['boundaries', 'boundaries.stylegan_bedroom256', 'boundaries.stylegan_ffhq256', 'boundaries.stylegan_tower256', 'models', 'utils'],
    'neural_style': [],
    'PhotoSketch': ['data', 'models', 'options', 'scripts', 'util'],
    'semantic-segmentation-pytorch': ['config', 'data', 'mit_semseg', 'mit_semseg.config', 'mit_semseg.lib', 'mit_semseg.lib.nn', 'mit_semseg.lib.nn.modules', 'mit_semseg.lib.nn.modules.tests', 'mit_semseg.lib.nn.parallel', 'mit_semseg.lib.utils', 'mit_semseg.lib.utils.data', 'mit_semseg.models'],
    'SPADE': ['data', 'datasets', 'models', 'models.networks', 'models.networks.sync_batchnorm', 'options', 'trainers', 'util'],
    'stylegan2': ['dnnlib', 'dnnlib.tflib', 'dnnlib.tflib.ops', 'dnnlib.submission', 'dnnlib.submission.internal', 'metrics', 'training'],
    'stylegan2-ada-pytorch': ['dnnlib', 'metrics', 'torch_utils', 'torch_utils.ops', 'training'],
    'tacotron2': ['text', 'waveglow'],
    'torch-dreams': ['torch_dreams'],
    'Wav2Lip': ['evaluation', 'evaluation.scores_LSE', 'face_detection', 'face_detection.detection', 'face_detection.detection.sfd', 'models'],
    'White-box-Cartoonization': ['index_files', 'test_code', 'test_code.saved_models', 'train_code', 'train_code.selective_search']
}


install_requires = [
    'bs4', 
    'dill', 
    'imutils',
    'inflect',
    'face_recognition', 
    'gdown',
    'ipython',
    'ipywidgets',
    'librosa',
    'lxml', 
    'matplotlib',
    'moviepy',
    'ninja',
    'noise', 
    'numba',
    'numpy',
    'opencv-python',
    'Pillow',
    'psutil',
    'scikit-image', 
    'scikit-learn', 
    'tensorflow-gpu==1.15.0',
    'torch', 
    'torchvision', 
    'tqdm',
    'unidecode',
    'yacs',
    "tqdm"
]

package_data = {
    'ml4a': [
        'models/submodules/stylegan2/dnnlib/tflib/ops/*.cu',
        'models/submodules/stylegan2-ada-pytorch/torch_utils/ops/*.cu',
        'models/submodules/stylegan2-ada-pytorch/torch_utils/ops/*.cpp',
        'models/submodules/stylegan2-ada-pytorch/torch_utils/ops/*.h'
        'models/submodules/face-parsing-PyTorch/modules/src/*.cu',
        'models/submodules/face-parsing-PyTorch/modules/src/*.cpp',
        'models/submodules/face-parsing-PyTorch/modules/src/*.h',
        'models/submodules/face-parsing-PyTorch/modules/src/utils/*.h',
        'models/submodules/face-parsing-PyTorch/modules/src/utils/*.cuh'
    ]
}


readme_file = pathlib.Path(__file__).parent / "README.md"

short_description = 'A toolkit for making art with machine learning, including an API for popular deep learning models, recipes for combining them, and a suite of educational examples'

for submodule, subfolders in submodules.items():
    submodule_packages = ['{}.{}'.format(submodules_root, submodule)]
    submodule_packages.extend(['{}.{}.{}'.format(submodules_root, submodule, f) for f in subfolders])
    packages.extend(submodule_packages)

setup(
    name='ml4a',
    version='0.1.2',
    description=short_description,
    long_description=readme_file.read_text(),
    long_description_content_type="text/markdown",
    url='http://github.com/ml4a/ml4a',
    author='Gene Kogan',
    author_email='gene@genekogan.com',
    license='MIT',
    packages=packages, 
    package_data=package_data,
    install_requires=install_requires,
    zip_safe=False
)
