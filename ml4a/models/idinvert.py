import os
import numpy as np

from ..utils import downloads
from . import submodules

with submodules.import_from('idinvert_pytorch'):
    from models import model_settings
    model_settings.MODEL_DIR = os.path.join(downloads.get_ml4a_downloads_folder(), 'idinvert_pytorch')
    from utils.editor import manipulate
    from utils.inverter import StyleGANInverter
    from models.helper import build_generator




model = None

# w.i.p



def build_inverter(model_name, num_iterations=100, regularization_loss_weight=2):
    inverter = StyleGANInverter(
        model_name,
        learning_rate=0.01,
        iteration=num_iterations,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=5e-5,
        regularization_loss_weight=regularization_loss_weight)
    return inverter


def fuse(inverter, context_images, target_image, size=256, crop_size=125, center_x=145, center_y=125):
    top = center_y - crop_size // 2
    left = center_x - crop_size // 2
    width, height = crop_size, crop_size

    if np.array(context_images).ndim < 4:
        context_images = [context_images]

    showed_fuses = []
    for context_image in context_images:
        mask_aug = np.ones((size, size, 1), np.uint8) * 255
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




def setup():
    global model
#     model_directory = downloads.download_from_gdrive(
#         '1TQf-LyS8rRDDapdcTnEgWzYJllPgiXdj', 
#         'photosketch/pretrained',
#         zip_file=True)

    
    # local repo files
    root = submodules.get_submodules_root('idinvert_pytorch')
    
    
    ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'expression']
    boundaries = {}
    for attr in ATTRS:
        boundary_path = os.path.join(os.path.join(root, 'boundaries'), 'stylegan_ffhq256', attr + '.npy')
        boundary_file = np.load(boundary_path, allow_pickle=True)[()]
        boundary = boundary_file['boundary']
        manipulate_layers = boundary_file['meta_data']['manipulate_layers']
        boundaries[attr] = []
        boundaries[attr].append(boundary)
        boundaries[attr].append(manipulate_layers)



    
    
    
    from ml4a import image
    from ml4a.utils import face


    model_name = 'styleganinv_ffhq256'
    inverter = build_inverter(model_name, num_iterations=100, regularization_loss_weight=0)


    img = image.load_image('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Vladimir_Putin_%282020-02-20%29.jpg/1200px-Vladimir_Putin_%282020-02-20%29.jpg')
    
    target_image, face_found = face.align_face(img, face_width=inverter.G.resolution)
    image.display(target_image, title="Display aligned face")


    context_paths = ['examples/000001.png', 'examples/000002.png', 'examples/000008.png',  'examples/000009.png']
    context_paths = [os.path.join(root, c) for c in context_paths]
    context_images = []
    for path in context_paths:
        img = image.load_image(path)
        aligned_face, face_found = face.align_face(img, face_width=inverter.G.resolution)
        if face_found:
            context_images.append(aligned_face)


    image.display(context_images)





    context_image = context_images[2]

    showed_fuse, diffused_image = fuse(inverter, 
                                       context_image, 
                                       target_image,
                                       size=inverter.G.resolution,
                                       crop_size=125,
                                       center_x=125,
                                       center_y=145)

    image.display(showed_fuse)
    image.display(diffused_image)


    #print('Building inverter')
    #inverter = build_inverter(model_name=model_name)
    print('Building generator')
    generator = build_generator(model_name)



    image_name = 'example.png'
    inverted_code_dir = 'here'
    #im_name = os.path.join(pre, image_name)
    #mani_image = aligned_img #align(inverter, im_name)



    latent_code_path = os.path.join(inverted_code_dir, image_name.split('.')[0] + '.npy')

    if not os.path.exists(latent_code_path):
        print('would go here')
        #latent_code, _ = invert(inverter, target_image)
        latent_code, reconstruction = inverter.easy_invert(target_image, num_viz=1)

    #np.save(latent_code_path, latent_code)
    else:
        print('code already exists, skip inversion!!!')
        latent_code = np.load(latent_code_path)



    print('Image inversion completed, please use the next block to manipulate!!!')

    
    #@title { display-mode: "form", run: "auto" }

    age = 2.0 #@param {type:"slider", min:-3.0, max:3.0, step:0.1}
    eyeglasses = 0 #@param {type:"slider", min:-2.9, max:3.0, step:0.1}
    gender = -1.0 #@param {type:"slider", min:-3.0, max:3.0, step:0.1}
    pose = 0 #@param {type:"slider", min:-3.0, max:3.0, step:0.1}
    expression = 0 #@param {type:"slider", min:-3.0, max:3.0, step:0.1}


    new_codes = latent_code.copy()
    for i, attr_name in enumerate(ATTRS):
        manipulate_layers = boundaries[attr_name][1]
        new_codes[:, manipulate_layers, :] += boundaries[attr_name][0][:, manipulate_layers, :] * eval(attr_name)

    new_images = generator.easy_synthesize(new_codes, **{'latent_space_type': 'wp'})['image']
    showed_images = np.concatenate([target_image[np.newaxis], new_images], axis=0)
    image.display(showed_images, num_cols=showed_images.shape[0])
    
    
    
    
def run(img):
    print('run')