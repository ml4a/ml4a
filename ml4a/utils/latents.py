import numpy as np


def interpolation_walk(endpoints, num_frames_per=30, loop=False):
    z1, z2 = endpoints[:-1], endpoints[1:]
    if loop:
        z1.append(endpoints[-1])
        z2.append(endpoints[0])
    z1, z2 = endpoints[:-1], endpoints[1:]
    if loop:
        z1.append(endpoints[-1])
        z2.append(endpoints[0])
    Z = np.concatenate([np.linspace(z_from, z_to, num_frames_per+1, axis=0)[:-1, :] 
                        for z_from, z_to in zip(z1, z2)], axis=0)
    Z = np.squeeze(Z)
    return Z
        