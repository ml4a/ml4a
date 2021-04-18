import os
import hashlib
from pathlib import Path
import numpy as np

from . import submodules
from ..utils import downloads

inverted_code_dir = os.path.join(
    downloads.get_ml4a_scratch_folder(), 
    'idinvert_pytorch/reconstructions'
)

model_dir = os.path.join(
    downloads.get_ml4a_downloads_folder(), 
    'idinvert_pytorch'
)

resolution = 256
    
models = {
    'bedroom': {
        'name': 'styleganinv_bedroom256',
        'inverter': '1ebuiaQ7xI99a6ZrHbxzGApEFCu0h0X2s',
        'generator': '1ka583QwvMOtcFZJcu29ee8ykZdyOCcMS',
        'attributes': {
            'cloth': '1PiOFd71eYTrJclwptYyxqbLBhUSQ0obT', 
            'cluttered_space': '1RBWZKE_NlI2cj4aG50VBVAFQZjthsEwL', 
            'indoor_lighting': '1z-egLTDGgJsHWiqCO2bQgFW4aHv2iYf5', 
            'scary': '1Bc19lhx4MQ_E9vGB02GRaFX7NpeNYPgd', 
            'soothing': '1s5vjjo3QbCYphaMOjsMwOLp9bzyO8S2E', 
            'wood': '1qOm1QehLJAeH2EQAPmufiVnmz7RWK-WN'
        }
    },
    'ffhq': {
        'name': 'styleganinv_ffhq256',
        'inverter': '1gij7xy05crnyA-tUTQ2F3yYlAlu6p9bO',
        'generator': '1SjWD4slw612z2cXa3-n38JwKZXqDUerG',
        'attributes': {
            'age': '1ez85GdHz9HZ6DgdQLMmji3ygdMoPipC-', 
            'expression': '1XJHe2gQKJEczBEhu2MGA94EX28AssnKM', 
            'eyeglasses': '1fFsNwMUUaPq_Hh6uPgA5K-v9Yjq8unjZ', 
            'gender': '1iWPlPYHl5h2UsB_ojqB8udJKqvn4y38w', 
            'pose': '1WSinkKoX9Y8xzfM0Ff2I6Jdum_nFgAy1'
        }
    },
    'tower': {
        'name': 'styleganinv_tower256',
        'inverter': '1Pzkgdi3xctdsCZa9lcb7dziA_UMIswyS',
        'generator': '1lI_OA_aN4-O3mXEPQ1Nv-6tdg_3UWcyN',
        'attributes': {
            'clouds': '18awC-Nq2Anx6qR-Kl2hteFxhQoo7vT9c', 
            'sunny': '1dZIG2UoXEszzySh1PP80Dlfi9XQJVeNJ', 
            'vegetation': '1LjhoneQ7vTXQ8lJb_CZeTF85ymtfwviB'
        }
    }
}

inverter = None
generator = None
attributes = None
current_model_name = None

cuda_available = submodules.cuda_available()

with submodules.localimport('submodules/idinvert_pytorch') as _importer:
    from models import model_settings
    model_settings.MODEL_DIR = model_dir
    from utils.editor import manipulate
    from utils.inverter import StyleGANInverter
    from models.helper import build_generator
    
        
def get_available_models():
    return models.keys()


def get_attributes():
    return attributes.keys()
    

def setup_model(model_name):
    global current_model_name
    if model_name == current_model_name:
        return
    
    assert model_name in get_available_models(), \
        'Error: {} not recognized. Available models are {}'.format(model_name, ', '.join(get_available_models()))
    
    # setup inverter and generator
    filename = models[model_name]['name'] 
    downloads.download_from_gdrive(
        models[model_name]['inverter'], 
        'idinvert_pytorch/{}_encoder.pth'.format(filename))
    downloads.download_from_gdrive(
        models[model_name]['generator'], 
        'idinvert_pytorch/{}_generator.pth'.format(filename))

    # setup attributes
    global attributes
    attributes = {}
    for attr_name in models[model_name]['attributes']:
        gdrive_id = models[model_name]['attributes'][attr_name]
        attr_path = 'idinvert_pytorch/attributes_{}/{}.npy'.format(model_name, attr_name)
        boundary_filename = downloads.download_from_gdrive(gdrive_id, attr_path)
        boundary_file = np.load(boundary_filename, allow_pickle=True)[()]
        boundary = boundary_file['boundary']
        manipulate_layers = boundary_file['meta_data']['manipulate_layers']
        attributes[attr_name] = [boundary, manipulate_layers]
    
    # setup VGG
    downloads.download_from_gdrive('1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y', 
                                   'idinvert_pytorch/vgg16.pth')
    
    current_model_name = model_name


def setup_inverter(model_name, num_iterations=1000, regularization_loss_weight=2):   
    global inverter
    setup_model(model_name)    
    inverter = StyleGANInverter(
        models[model_name]['name'],
        learning_rate=0.01,
        iteration=num_iterations,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=5e-5,
        regularization_loss_weight=regularization_loss_weight)

    
def setup_generator(model_name):
    global generator
    setup_model(model_name)
    generator = build_generator(models[model_name]['name'])

        
def fuse(model_name, context_images, target_image, crop_size=125, center_x=145, center_y=125):
    if not inverter or models[model_name]['name'] != inverter.model_name:
        setup_inverter(model_name)
    
    target_image = np.array(target_image)
    top = center_y - crop_size // 2
    left = center_x - crop_size // 2
    width, height = crop_size, crop_size

    if np.array(context_images).ndim < 4:
        context_images = [context_images]

    showed_fuses = []
    for context_image in context_images:
        mask_aug = np.ones((resolution, resolution, 1), np.uint8) * 255
        paste_image = np.array(context_image).copy()
        paste_image[top:top + height, left:left + width] = target_image[top:top + height, left:left + width].copy()
        showed_fuse = np.concatenate([paste_image, mask_aug], axis=2)
        showed_fuses.append(showed_fuse)

    _, diffused_images = inverter.easy_diffuse(target=target_image,
                                               context=np.array(context_images),
                                               center_x=center_x,
                                               center_y=center_y,
                                               crop_x=width,
                                               crop_y=height,
                                               num_viz=1)
        
    diffused_images = [np.concatenate([images[-1], mask_aug], axis=2) 
                       for key, images in diffused_images.items()]

    return showed_fuses, diffused_images


def invert(model_name, target_image, redo=False, save=False):
    if not inverter or models[model_name]['name'] != inverter.model_name:
        setup_inverter(model_name)
    
    target_image = np.array(target_image)
    
    image_hash  = hashlib.md5(target_image).hexdigest()
    image_hash += hashlib.md5(model_name.encode('utf-8')).hexdigest()
    
    latent_code_path = os.path.join(inverted_code_dir, image_hash+'.npy')
    latent_code_found = os.path.exists(latent_code_path)
    
    if not latent_code_found or redo:
        print('optimizing latent_code to reconstruct target image...')
        latent_code, reconstruction = inverter.easy_invert(target_image, num_viz=1)
    else:
        print('previous code found at {}, skip inversion (set redo=True to overwrite)'.format(latent_code_path))
        latent_code = np.load(latent_code_path)

    if save and (not latent_code_found or redo):
        print('saving latent code to {}'.format(latent_code_path))
        Path(inverted_code_dir).mkdir(parents=True, exist_ok=True)
        np.save(latent_code_path, latent_code)
            
    return latent_code


def generate(model_name, latent_code):
    if not generator or models[model_name]['name'] != generator.model_name:
        setup_generator(model_name)
    return generator.easy_synthesize(latent_code, **{'latent_space_type': 'wp'})['image']


def modulate(latent_code, attribute, amount):
    assert attributes, "Error: no model loaded!"    
    assert attribute in attributes.keys(), \
        'attribute {} not found. available: {}'.format(attribute, ', '.join(attributes))
    new_code = latent_code.copy()
    manipulate_layers = attributes[attribute][1]
    new_code[:, manipulate_layers, :] += attributes[attribute][0][:, manipulate_layers, :] * amount
    return new_code

