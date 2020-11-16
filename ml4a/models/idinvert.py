import os
import hashlib
from pathlib import Path
import numpy as np

from . import submodules
from ..utils import downloads

inverted_code_dir = os.path.join(downloads.get_ml4a_downloads_folder(), 'idinvert_pytorch/reconstructions')
model_dir = os.path.join(downloads.get_ml4a_downloads_folder(), 'idinvert_pytorch/pretrained')

available_models = ['styleganinv_ffhq256', 'styleganinv_tower256', 'styleganinv_bedroom256']
attributes = ['age', 'eyeglasses', 'gender', 'pose', 'expression']
resolution = 256

inverter = None
generator = None
boundaries = None

with submodules.localimport('submodules/idinvert_pytorch') as _importer:
    from models import model_settings
    model_settings.MODEL_DIR = model_dir
    from utils.editor import manipulate
    from utils.inverter import StyleGANInverter
    from models.helper import build_generator

    
def setup_boundary_vectors():
    global boundaries
    root = submodules.get_submodules_root('idinvert_pytorch')
    boundary_folder = os.path.join(root, 'boundaries')    
    boundaries = {}
    for attr in attributes:
        boundary_path = os.path.join(boundary_folder, 'stylegan_ffhq256', attr + '.npy')
        boundary_file = np.load(boundary_path, allow_pickle=True)[()]
        boundary = boundary_file['boundary']
        manipulate_layers = boundary_file['meta_data']['manipulate_layers']
        boundaries[attr] = [boundary, manipulate_layers]


def download_pretrained_model(model_name):
    if model_name == 'styleganinv_ffhq256':
        downloads.download_from_gdrive('1gij7xy05crnyA-tUTQ2F3yYlAlu6p9bO', 
                                       'idinvert_pytorch/pretrained/styleganinv_ffhq256_encoder.pth')
        downloads.download_from_gdrive('1SjWD4slw612z2cXa3-n38JwKZXqDUerG', 
                               'idinvert_pytorch/pretrained/styleganinv_ffhq256_generator.pth')

    elif model_name == 'styleganinv_tower256':
        downloads.download_from_gdrive('1Pzkgdi3xctdsCZa9lcb7dziA_UMIswyS', 
                                       'idinvert_pytorch/pretrained/styleganinv_tower256_encoder.pth')
        downloads.download_from_gdrive('1lI_OA_aN4-O3mXEPQ1Nv-6tdg_3UWcyN', 
                                       'idinvert_pytorch/pretrained/styleganinv_tower256_generator.pth')

    elif model_name == 'styleganinv_bedroom256':
        downloads.download_from_gdrive('1ebuiaQ7xI99a6ZrHbxzGApEFCu0h0X2s', 
                                       'idinvert_pytorch/pretrained/styleganinv_bedroom256_encoder.pth')
        downloads.download_from_gdrive('1ka583QwvMOtcFZJcu29ee8ykZdyOCcMS', 
                                       'idinvert_pytorch/pretrained/styleganinv_bedroom256_generator.pth')

    downloads.download_from_gdrive('1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y', 
                                   'idinvert_pytorch/pretrained/vgg16.pth')


def setup_inverter(model_name, num_iterations=100, regularization_loss_weight=2):   
    global inverter
    download_pretrained_model(model_name)
    inverter = StyleGANInverter(
        model_name,
        learning_rate=0.01,
        iteration=num_iterations,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=5e-5,
        regularization_loss_weight=regularization_loss_weight)

    
def setup_generator(model_name):
    global generator
    download_pretrained_model(model_name)
    generator = build_generator(model_name)

        
def fuse(model_name, context_images, target_image, crop_size=125, center_x=145, center_y=125):
    if not inverter or model_name != inverter.model_name:
        setup_inverter(model_name)
        
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
    if not inverter or model_name != inverter.model_name:
        setup_inverter(model_name)

    image_hash = hashlib.md5(target_image).hexdigest()
    
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
    if not generator or model_name != generator.model_name:
        setup_generator(model_name)

    return generator.easy_synthesize(latent_code, **{'latent_space_type': 'wp'})['image']


def modify_latent_code(latent_code, attribute, amount):
    if attribute not in attributes:
        print('attribute %s not found. available: {}'.format(', '.join(attributes)))
    
    if not boundaries:
        setup_boundary_vectors()

    new_code = latent_code.copy()
    manipulate_layers = boundaries[attribute][1]
    new_code[:, manipulate_layers, :] += boundaries[attribute][0][:, manipulate_layers, :] * amount
    return new_code


def age(latent_code, amount):
    return modify_latent_code(latent_code, 'age', amount)

def eyeglasses(latent_code, amount):
    return modify_latent_code(latent_code, 'eyeglasses', amount)

def gender(latent_code, amount):
    return modify_latent_code(latent_code, 'gender', amount)

def pose(latent_code, amount):
    return modify_latent_code(latent_code, 'pose', amount)

def expression(latent_code, amount):
    return modify_latent_code(latent_code, 'expression', amount)

