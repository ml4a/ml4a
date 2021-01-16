import numpy as np
import tensorflow as tf
from threading import Lock

from ..utils import downloads
from .. import image
from . import submodules

cuda_available = submodules.cuda_available()

with submodules.localimport('submodules/glow') as _importer:
    pass

# todo: wrong version of blocksparse doesn't let you use optimized pb

lock = Lock()
sess, update_feed = None, None
eps_size, eps_shapes, n_eps = None, None, None
enc_eps, dec_eps, enc_x, dec_x = None, None, None, None

_TAGS = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split()



def get_attributes():
    return _TAGS


def get_tensor(name):
    return tf.compat.v1.get_default_graph().get_tensor_by_name('import/' + name + ':0')


def run(sess, fetches, feed_dict):
    with lock:
        # Locked tensorflow so average server response time to user is lower
        result = sess.run(fetches, feed_dict)
    return result


def flatten_eps(eps):
    # [BS, eps_size]
    return np.concatenate([np.reshape(e, (e.shape[0], -1)) for e in eps], axis=-1)


def unflatten_eps(feps):
    index = 0
    eps = []
    bs = feps.shape[0]  # feps.size // eps_size
    for shape in eps_shapes:
        eps.append(np.reshape(
            feps[:, index: index+np.prod(shape)], (bs, *shape)))
        index += np.prod(shape)
    return eps


def encode(img):
    img = np.array(img)
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    bs = img.shape[0]
    assert img.shape[1:] == (256, 256, 3)
    feed_dict = {enc_x: img}
    update_feed(feed_dict, bs)  # For unoptimized model
    return flatten_eps(run(sess, enc_eps, feed_dict))


def decode(feps):
    feps = np.array(feps)
    if len(feps.shape) == 1:
        feps = np.expand_dims(feps, 0)
    bs = feps.shape[0]
    eps = unflatten_eps(feps)
    
    #assert len(eps) == n_eps
    #for i in range(n_eps):
    #   shape = (BATCH_SIZE, 128 // (2 ** i), 128 // (2 ** i), 6 * (2 ** i) * (2 ** (i == (n_eps - 1))))
    #   assert eps[i].shape == shape

    feed_dict = {}
    for i in range(n_eps):
        feed_dict[dec_eps[i]] = eps[i]

    update_feed(feed_dict, bs)  # For unoptimized model
    return run(sess, dec_x, feed_dict)


def project(z):
    return np.dot(z, z_proj)


def _manipulate(z, dz, alpha):
    z = z + alpha * dz
    return decode(z), z


def _manipulate_range(z, dz, points, scale):
    z_range = np.concatenate(
        [z + scale*(pt/(points - 1)) * dz for pt in range(0, points)], axis=0)
    return decode(z_range), z_range


def mix(z1, z2, alpha): # alpha goes from [0, 1]
    dz = (z2 - z1)
    return _manipulate(z1, dz, alpha)


def mix_range(z1, z2, points=5):
    dz = (z2 - z1)
    return _manipulate_range(z1, dz, points, 1.)


def manipulate(z, typ, alpha): # alpha goes from [-1, 1]
    dz = z_manipulate[typ] 
    return _manipulate(z, dz, alpha)


def manipulate_all(z, typs, alphas):
    dz = 0.0
    for i in range(len(typs)):
        dz += alphas[i] * z_manipulate[typs[i]]
    return _manipulate(z, dz, 1.0)


def manipulate_range(z, typ, points=5, scale=1):
    dz = z_manipulate[typ]
    return _manipulate_range(z - dz, 2*dz, points, scale)


def random(bs=1, eps_std=0.7):
    feps = np.random.normal(scale=eps_std, size=[bs, eps_size])
    return decode(feps), feps


def add_attribute(z, z_direction, amt):
    if isinstance(z_direction, str):
        assert z_direction in _TAGS, "Error: %s not a known attribute. Look up glow.get_attributes()." % z_direction
        z_direction = z_manipulate[_TAGS.index(z_direction)]
    z_new = np.copy(z)
    z_new[0] += amt * z_direction
    return z_new


def load_model(optimized):

    global sess, update_feed
    global eps_size, eps_shapes, n_eps
    global enc_eps, dec_eps, enc_x, dec_x
    global z_manipulate

    graph_optimized_path = downloads.download_data_file(
        url='https://storage.googleapis.com/glow-demo/large3/graph_optimized.pb', 
        output_path='glow/data/graph_optimized.pb')
    graph_unoptimized_path = downloads.download_data_file(
        url='https://storage.googleapis.com/glow-demo/large3/graph_unoptimized.pb', 
        output_path='glow/data/graph_unoptimized.pb')
    z_manipulate_path = downloads.download_data_file(
        url='https://storage.googleapis.com/glow-demo/z_manipulate.npy', 
        output_path='glow/data/z_manipulate.npy')

    if optimized:
        # Optimized model. Twice as fast as
        # 1. we freeze conditional network (label is always 0)
        # 2. we use fused kernels
        import blocksparse
        graph_path = graph_optimized_path
        inputs = {
            'dec_eps_0': 'dec_eps_0',
            'dec_eps_1': 'dec_eps_1',
            'dec_eps_2': 'dec_eps_2',
            'dec_eps_3': 'dec_eps_3',
            'dec_eps_4': 'dec_eps_4',
            'dec_eps_5': 'dec_eps_5',
            'enc_x': 'input/enc_x',
        }
        outputs = {
            'dec_x': 'model_3/Cast_1',
            'enc_eps_0': 'model_2/pool0/truediv_1',
            'enc_eps_1': 'model_2/pool1/truediv_1',
            'enc_eps_2': 'model_2/pool2/truediv_1',
            'enc_eps_3': 'model_2/pool3/truediv_1',
            'enc_eps_4': 'model_2/pool4/truediv_1',
            'enc_eps_5': 'model_2/truediv_4'
        }

        def update_feed(feed_dict, bs):
            return feed_dict
    else:
        graph_path = graph_unoptimized_path
        inputs = {
            'dec_eps_0': 'Placeholder',
            'dec_eps_1': 'Placeholder_1',
            'dec_eps_2': 'Placeholder_2',
            'dec_eps_3': 'Placeholder_3',
            'dec_eps_4': 'Placeholder_4',
            'dec_eps_5': 'Placeholder_5',
            'enc_x': 'input/image',
            'enc_x_d': 'input/downsampled_image',
            'enc_y': 'input/label'
        }
        outputs = {
            'dec_x': 'model_1/Cast_1',
            'enc_eps_0': 'model/pool0/truediv_1',
            'enc_eps_1': 'model/pool1/truediv_1',
            'enc_eps_2': 'model/pool2/truediv_1',
            'enc_eps_3': 'model/pool3/truediv_1',
            'enc_eps_4': 'model/pool4/truediv_1',
            'enc_eps_5': 'model/truediv_4'
        }

        def update_feed(feed_dict, bs):
            x_d = 128 * np.ones([bs, 128, 128, 3], dtype=np.uint8)
            y = np.zeros([bs], dtype=np.int32)
            feed_dict[enc_x_d] = x_d
            feed_dict[enc_y] = y
            return feed_dict

    with tf.io.gfile.GFile(graph_path, 'rb') as f:
        graph_def_optimized = tf.compat.v1.GraphDef()
        graph_def_optimized.ParseFromString(f.read())

    # Init session and params
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(0)
    sess = tf.compat.v1.Session(config=config)

    # Import Graph definition
    tf.import_graph_def(graph_def_optimized)
    print("Loaded model")

    n_eps = 6

    # Encoder
    enc_x = get_tensor(inputs['enc_x'])
    enc_eps = [get_tensor(outputs['enc_eps_' + str(i)]) for i in range(n_eps)]
    if not optimized:
        enc_x_d = get_tensor(inputs['enc_x_d'])
        enc_y = get_tensor(inputs['enc_y'])

    # Decoder
    dec_x = get_tensor(outputs['dec_x'])
    dec_eps = [get_tensor(inputs['dec_eps_' + str(i)]) for i in range(n_eps)]
    eps_shapes = [(128, 128, 6), (64, 64, 12), (32, 32, 24),
                (16, 16, 48), (8, 8, 96), (4, 4, 384)]
    eps_sizes = [np.prod(e) for e in eps_shapes]
    eps_size = 256 * 256 * 3
    z_manipulate = np.load(z_manipulate_path)

    flip_tags = ['No_Beard', 'Young']
    for tag in flip_tags:
        i = _TAGS.index(tag)
        z_manipulate[i] = -z_manipulate[i]

    scale_tags = ['Narrow_Eyes']
    for tag in scale_tags:
        i = _TAGS.index(tag)
        z_manipulate[i] = 1.2*z_manipulate[i]

    z_sq_norms = np.sum(z_manipulate**2, axis=-1, keepdims=True)
    z_proj = (z_manipulate / z_sq_norms).T


def warm_start():
    _img, _z = random(1)
    _z = encode(_img)
    print("Warm started GLOW model")


# def test():
#     import time
#     img = Image.open('test/img.png')
#     img = np.reshape(np.array(img), [1, 256, 256, 3])

#     # Encoding speed
#     eps = encode(img)
#     t = time.time()
#     for _ in tqdm(range(10)):
#         eps = encode(img)
#     print("Encoding latency {} sec/img".format((time.time() - t) / (1 * 10)))

#     # Decoding speed
#     dec = decode(eps)
#     t = time.time()
#     for _ in tqdm(range(10)):
#         dec = decode(eps)
#     print("Decoding latency {} sec/img".format((time.time() - t) / (1 * 10)))
#     img = Image.fromarray(dec[0])
#     #img.save('test/dec.png')

#     # Manipulation
#     dec, _ = manipulate(eps, _TAGS.index('Smiling'), 0.66)
#     img = Image.fromarray(dec[0])
#     #img.save('test/smile.png')
