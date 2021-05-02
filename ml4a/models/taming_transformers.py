import os
import yaml
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from ..utils import downloads
from .. import image
from . import submodules

with submodules.import_from('taming-transformers'):  # localimport fails here
    import taming.modules.losses
    from taming.models.vqgan import VQModel
    from taming.models.cond_transformer import Net2NetTransformer


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = None
config = None

pretrained_models = {
    'net2net': {
        'checkpoint': 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1',
        'config': 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1'
    },
    'vqgan': {
        'checkpoint': 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1',
        'config': 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1'
    }
}


def get_pretrained_models():
    return pretrained_models.keys()


def setup(model_name):
    global model, config
    
    assert model_name in get_pretrained_models(), \
        'Error: {} not recognized checkpoint'.format(model_name)

    checkpoint = downloads.download_data_file(
        pretrained_models[model_name]['checkpoint'],
        'taming-transformers/{}/checkpoint.ckpt'.format(model_name))

    config_file = downloads.download_data_file(
        pretrained_models[model_name]['config'],
        'taming-transformers/{}/config.yaml'.format(model_name))

    config = OmegaConf.load(config_file)
    #print(yaml.dump(OmegaConf.to_container(config)))
    if model_name == 'net2net':
        model = Net2NetTransformer(**config.model.params)
    elif model_name == 'vqgan':
        model = VQModel(**config.model.params)
    sd = torch.load(checkpoint, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)

    #model.cuda().eval()
    model = model.eval()
    model = model.to(DEVICE)
    #torch.set_grad_enabled(False)
    

def get_example_segmentation():
    segmentation_path = downloads.download_data_file(
        'https://github.com/CompVis/taming-transformers/raw/master/data/sflckr_segmentations/norway/25735082181_999927fe5a_b.png',
        'taming-transformers/{}/25735082181_999927fe5a_b.png'.format('net2net'))
    segmentation = Image.open(segmentation_path)    
    return segmentation


def generate_from_segmentation(segmentation):
    assert model is not None, 'Error: no model loaded'
        
    segmentation = np.array(segmentation)
    segmentation = np.eye(182)[segmentation]
    segmentation = torch.tensor(segmentation.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)

    c_code, c_indices = model.encode_to_c(segmentation)
    assert c_code.shape[2]*c_code.shape[3] == c_indices.shape[1]

    segmentation_rec = model.cond_stage_model.decode(c_code)
    output = torch.softmax(segmentation_rec, dim=1)
    return output
#    show_segmentation(output)
    

def run2():
    codebook_size = config.model.params.first_stage_config.params.embed_dim
    z_indices_shape = c_indices.shape
    z_code_shape = c_code.shape
    z_indices = torch.randint(codebook_size, z_indices_shape, device=model.device)
    x_sample = model.decode_to_img(z_indices, z_code_shape)
    show_image(x_sample)
    
    
    from IPython.display import clear_output
    import time

    idx = z_indices
    idx = idx.reshape(z_code_shape[0],z_code_shape[2],z_code_shape[3])

    cidx = c_indices
    cidx = cidx.reshape(c_code.shape[0],c_code.shape[2],c_code.shape[3])

    temperature = 1.0
    top_k = 100
    update_every = 50

    start_t = time.time()
    for i in range(0, z_code_shape[2]-0):
        if i <= 8:
            local_i = i
        elif z_code_shape[2]-i < 8:
            local_i = 16-(z_code_shape[2]-i)
        else:
            local_i = 8
        for j in range(0,z_code_shape[3]-0):
            if j <= 8:
                local_j = j
            elif z_code_shape[3]-j < 8:
                local_j = 16-(z_code_shape[3]-j)
            else:
                local_j = 8

        i_start = i-local_i
        i_end = i_start+16
        j_start = j-local_j
        j_end = j_start+16

        patch = idx[:,i_start:i_end,j_start:j_end]
        patch = patch.reshape(patch.shape[0],-1)
        cpatch = cidx[:, i_start:i_end, j_start:j_end]
        cpatch = cpatch.reshape(cpatch.shape[0], -1)
        patch = torch.cat((cpatch, patch), dim=1)
        logits,_ = model.transformer(patch[:,:-1])
        logits = logits[:, -256:, :]
        logits = logits.reshape(z_code_shape[0],16,16,-1)
        logits = logits[:,local_i,local_j,:]

        logits = logits/temperature

        if top_k is not None:
            logits = model.top_k_logits(logits, top_k)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx[:,i,j] = torch.multinomial(probs, num_samples=1)

        step = i*z_code_shape[3]+j
        if step%update_every==0 or step==z_code_shape[2]*z_code_shape[3]-1:
            x_sample = model.decode_to_img(idx, z_code_shape)
            clear_output()
            print(f"Time: {time.time() - start_t} seconds")
            print(f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
            show_image(x_sample)


def show_segmentation(segmentation):
    assert model is not None, 'Error: no model loaded'

    segmentation = np.array(segmentation)
    segmentation = np.eye(182)[segmentation]
    segmentation = torch.tensor(segmentation.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)
    s = segmentation.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,None,:]
    colorize = np.random.RandomState(1).randn(1,1,s.shape[-1],3)
    colorize = colorize / colorize.sum(axis=2, keepdims=True)
    s = s@colorize
    s = s[...,0,:]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    s = Image.fromarray(s)
    image.display(s)
    
# def get_output(s):
#     s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
#     s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
#     s = Image.fromarray(s)
#     return image.load_image(s)
