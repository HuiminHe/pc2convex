import os
import numpy as np
from body_scan import BodyScan
from time import time

'''
utility functions
'''

def get_name(fpath):
    '''
    get file name without extension from fpath
    :param fpath:
    :return:
    A file name string
    '''
    _, fname = os.path.split(fpath)
    fname, _ = os.path.splitext(fname)
    return fname


def read_data(fpath):
    '''
    read a3d data from fpaht
    :param fpath: file path
    :return:
    An numpy n-dimensional array of 512x512x660
    '''
    tic = time()
    bs = BodyScan(fpath)
    data, _ = bs.read_img_data()
    toc = time()
    print('read {} costs {}s'.format(fpath, toc-tic))
    return data


def get_points(image, thresh=0):
    indx = np.where(image > thresh)
    return np.vstack(indx).T.astype(np.int16)


def pc2img(data, threshold=0.1, axis=0):
    img = np.amax(data, axis)
    max_ = np.amax(img)
    min_ = np.amin(img)
    img = np.clip((img - min_) / (max_ - min_) - threshold, 0, None)
    img = img.transpose()
    img = np.flipud(img)
    return img
