from PIL import Image
import numpy as np
from tqdm import tqdm



def histogram_equalization(x):
    hist, bins = np.histogram(x.flatten(), 255, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255.0/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    x2 = cdf[x.astype('uint8')]
    return x2, cdf


def get_histogram(pixels, bright=True):
    hist = np.zeros((256,1 if bright else 3))
    for p in pixels:
        if bright:
            avg = int((p[0]+p[1]+p[2])/3.0)
            hist[avg,0] += 1
        else:
            hist[p[0],0] += 1
            hist[p[1],1] += 1
            hist[p[2],2] += 1
    return np.array(hist)


def match_histogram2(img1, hist):
    colors = img1.getdata()
    red, green, blue = [c[0] for c in colors], [c[1] for c in colors], [c[2] for c in colors]
    sr = sorted(range(len(red)), key=lambda k: red[k])
    sg = sorted(range(len(green)), key=lambda k: green[k])
    sb = sorted(range(len(blue)), key=lambda k: blue[k])
    hr, hg, hb = [[hist[i][c] for i in range(256)] for c in range(3)]
    fr, fg, fb = 0,0,0
    for c in range(len(hr)):
        nfr, nfg, nfb = int(hr[c]), int(hg[c]), int(hb[c])
        idxr = [sr[k] for k in range(fr,fr+nfr)]
        idxg = [sg[k] for k in range(fg,fg+nfg)]
        idxb = [sb[k] for k in range(fb,fb+nfb)]
        for ir in idxr:
            red[ir] = c
        for ig in idxg:
            green[ig] = c
        for ib in idxb:
            blue[ib] = c
        fr, fg, fb = fr+nfr, fg+nfg, fb+nfb
    adjusted_colors = zip(red, green, blue)
    img_adjusted = Image.new(img1.mode, img1.size)
    img_adjusted.putdata(adjusted_colors)
    return img_adjusted


def match_histogram(img1, hist):
    pixels = list(img1.getdata())
    red, green, blue = np.array([c[0] for c in pixels]), np.array([c[1] for c in pixels]), np.array([c[2] for c in pixels])
    sr = sorted(range(len(red)), key=lambda k: red[k])
    sg = sorted(range(len(green)), key=lambda k: green[k])
    sb = sorted(range(len(blue)), key=lambda k: blue[k])
    num_pixel_mult = (3 * len(pixels)) / np.sum(hist)
    hr, hg, hb = [[int(num_pixel_mult * hist[i][c]) for i in range(256)] for c in range(3)]
    fr, fg, fb = 0, 0, 0
    for c in range(len(hr)):
        nfr, nfg, nfb = int(hr[c]), int(hg[c]), int(hb[c])
        red[np.array([sr[k] for k in xrange(fr,fr+nfr)]).astype('int')] = c
        green[np.array([sg[k] for k in xrange(fg,fg+nfg)]).astype('int')] = c
        blue[np.array([sb[k] for k in xrange(fb,fb+nfb)]).astype('int')] = c
        fr, fg, fb = fr+nfr, fg+nfg, fb+nfb
    adjusted_pixels = zip(red, green, blue)
    img_adjusted = Image.new(img1.mode, img1.size)
    img_adjusted.putdata(adjusted_pixels)
    return img_adjusted


def adjust_color_range(img, hist, amt, border):
    cdf = hist.cumsum() / np.sum(hist)
    i1, i2 = min([i for i in range(256) if cdf[i]>border]), max([i for i in range(256) if cdf[i]<1.0-border])
    j1, j2 = int((1.0-amt)*i1), i2 + amt*(255-i2)
    img2 = np.clip(j1 + (j2-j1)*(img - i1)/(i2-i1), 0.0, 255.0)
    return img2


def get_average_histogram(frames_path):
    numframes = len([f for f in listdir(frames_path) if isfile(join(frames_path, f)) and f[-4:]=='.png'])
    img = Image.open('%s/f00001.png'%(frames_path))
    histogram = get_histogram(list(img.getdata()))
    for t in tqdm(range(1,numframes,8)):
        img = Image.open('%s/f%05d.png'%(frames_path, t+1))
        histogram += get_histogram(list(img.getdata()))
    histogram /= (1+len(range(1,numframes,8)))
    return histogram
