import numpy as np
import torch
import torchvision.models as models
from . import submodules
from .. import image
from ..utils import EasyDict

with submodules.localimport('submodules/torch-dreams') as _importer:
    from torch_dreams.dreamer import dreamer
    from torch_dreams.custom_image_param import custom_image_param
    from torch_dreams import auto_image_param


model = None
dreamy_boi = None
layer_lookup = None


def setup_torch_dreams():
    global model, dreamy_boi
    model = models.inception_v3(pretrained=True)
    dreamy_boi = dreamer(model, device = 'cuda', quiet =  False)
    setup_layer_lookup()
    return dreamy_boi


def make_custom_func(layer_number=0, channel_number=0): 
    def custom_func(layer_outputs):
        loss = layer_outputs[layer_number][channel_number].mean()
        return -loss
    return custom_func


def run(config, img, title=None):
    if not dreamy_boi or not model:
        setup_torch_dreams()

    # get config 
    assert 'objective' in config, 'No objective to optimize in config'
    c = EasyDict(config)
    width, height = c.size if 'size' in c else (256, 256)
    iters = c.iters if 'iters' in c else 150
    lr = c.lr if 'lr' in c else 9e-3
    rotate_degrees = c.rotate_degrees if 'rotate_degrees' in c else 15
    scale_max = c.scale_max if 'scale_max' in c else 1.2
    scale_min = c.scale_min if 'scale_min' in c else 0.5
    translate = c.translate if 'translate' in c else (0.2, 0.2)
    weight_decay = c.weight_decay if 'weight_decay' in c else 1e-2
    grad_clip = c.grad_clip if 'grad_clip' in c else 1
    layer = get_layer(c.objective['layer'])
    channel = c.objective['channel']
    
    # set target
    layers_to_use = [layer]
    my_custom_func = make_custom_func(
        layer_number = 0, 
        channel_number = channel)

    if img is None:
        input_param = auto_image_param(height=height, width=width, device='cuda', standard_deviation=0.01)
    else:
        if isinstance(img, str):
            img = image.load_image(img)
        img = image.resize(img, (width, height))
        img = torch.tensor(np.array(img)/255).permute(-1,0,1).unsqueeze(0)
        input_param = custom_image_param(image=img, device= 'cuda')

    # run torch_dreams
    output_param = dreamy_boi.render(        
        image_parameter = input_param,        
        layers = layers_to_use,
        custom_func = my_custom_func,
        width = width, 
        height = height,
        iters = iters,
        lr = lr,
        rotate_degrees = rotate_degrees,
        scale_max = scale_max,
        scale_min = scale_min,
        translate_x = translate[0],
        translate_y = translate[1],
        weight_decay = weight_decay,
        grad_clip = grad_clip
    )

    output_image = (255*np.clip(np.array(output_param), 0, 1)).astype(np.uint8)
    return output_image


def get_layer(query):
    if not model or not layer_lookup:
        setup_torch_dreams()
    if query in layer_lookup:
        return layer_lookup[query]
    else:
        print('Layer {} not found'.format(query))
        return None
    
    
def setup_layer_lookup():
    global layer_lookup
    if not model:
        setup_torch_dreams()
    layer_lookup = {}
    keys = list(dict(model.named_children()).keys())
    values = list(dict(model.named_children()).values())
    for k, v in zip(keys, values):
        keys2 = list(dict(v.named_children()).keys())
        values2 = list(dict(v.named_children()).values())
        layer_lookup['{}'.format(k)] = v
        for k2, v2 in zip(keys2, values2):
            keys3 = list(dict(v2.named_children()).keys())
            values3 = list(dict(v2.named_children()).values())
            layer_lookup['{}.{}'.format(k, k2)] = v2
            for k3, v3 in zip(keys3, values3):
                layer_lookup['{}.{}.{}'.format(k, k2, k3)] = v3


def print_layers():
    if not model:
        setup_torch_dreams()
    keys = list(dict(model.named_children()).keys())
    values = list(dict(model.named_children()).values())
    for k, v in zip(keys, values):
        keys2 = list(dict(v.named_children()).keys())
        values2 = list(dict(v.named_children()).values())
        if isinstance(v, torch.nn.Conv2d):
            print('{} ({})'.format(k, v.out_channels))
        else:
            print('{}'.format(k))        
        for k2, v2 in zip(keys2, values2):
            keys3 = list(dict(v2.named_children()).keys())
            values3 = list(dict(v2.named_children()).values())
            if isinstance(v2, torch.nn.Conv2d):
                print('  {}.{} ({})'.format(k, k2, v2.out_channels))
            else:
                print('  {}.{}'.format(k, k2))
            for k3, v3 in zip(keys3, values3):
                if isinstance(v3, torch.nn.Conv2d):
                    print('    {}.{}.{} ({})'.format(k, k2, k3, v3.out_channels))
                else:
                    print('    {}.{}.{}'.format(k, k2, k3))
 
                
