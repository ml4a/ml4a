from ..utils import downloads
from . import submodules

with submodules.import_from('neural-synth'):
    from model import *
    from dream import *

    
def test():

    params = DeepDreamArgs()
    params.tile_size = 512

#     params.model_file = 'model/tensorflow_inception_graph.pb'

    
    params.model_file = downloads.download_from_gdrive(
        '1G7_wifUk8HRjFIfYZb-A6lpPV_azld06', 
        'deepdream/tensorflow_inception_graph.pb')

    deepdream = DeepDream(params)



    config = {
        'objective': [
            {'layer': 'mixed4c_pool_reduce', 'channel': 61},
            {'layer': 'mixed4d_3x3_bottleneck_pre_relu', 'channel': 22}
        ],
        'num_octaves': 5,
        'octave_ratio': 1.333,
        'num_iterations': 32,
        'lap_n': 5,
        'masks': ['../../../neural-style-pt/images/masks/monalisa1a.png',
                  '../../../neural-style-pt/images/masks/monalisa1b.png'],
        'step': 1.25,
        'size': 512,
        'grayscale_gradients': False,
        'normalize_gradients': True
    }


    img = load_image('../../../neural-style-pt/images/inputs/monalisa.jpg', 64) #random_image((2048, 2048))
    img = run_deepdream(deepdream, config, img)
    display(img)
    save(img, 'output.png')
