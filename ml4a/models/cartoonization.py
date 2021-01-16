import os
import numpy as np
import tensorflow as tf 
from tqdm import tqdm

from ..utils import downloads
from . import submodules

cuda_available = submodules.cuda_available()

with submodules.localimport('submodules/White-box-Cartoonization') as _importer:
    from test_code import network
    from test_code import guided_filter

sess = None


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
    
    
def setup():
    global sess, input_photo, final_out
    
    model_subfolder = 'White-box-Cartoonization/saved_model'
    model_fullpath = os.path.join(downloads.get_ml4a_downloads_folder(), model_subfolder)
    
    downloads.download_data_file(
        'https://raw.githubusercontent.com/SystemErrorWang/White-box-Cartoonization/master/test_code/saved_models/model-33999.data-00000-of-00001', 
        os.path.join(model_subfolder, 'model-33999.data-00000-of-00001')
    )
    downloads.download_data_file(
        'https://raw.githubusercontent.com/SystemErrorWang/White-box-Cartoonization/master/test_code/saved_models/model-33999.index', 
        os.path.join(model_subfolder, 'model-33999.index')
    )
    downloads.download_text_file(
        'https://raw.githubusercontent.com/SystemErrorWang/White-box-Cartoonization/master/test_code/saved_models/checkpoint', 
        os.path.join(model_subfolder, 'checkpoint')
    )

    input_photo = tf.placeholder(tf.float32, [None, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.compat.v1.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.compat.v1.train.Saver(var_list=gene_vars)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    saver.restore(sess, tf.train.latest_checkpoint(model_fullpath))

    
def run(img):
    if sess is None:
        setup()

    img = np.array(img)
    #img = resize_crop(img)    
    img = img.astype(np.float32)/127.5 - 1
    if img.ndim < 4:
        img = np.expand_dims(img, axis=0)
    output = sess.run(final_out, feed_dict={input_photo: img})
    output = (np.squeeze(output)+1)*127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

    
