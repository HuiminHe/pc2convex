import os
import numpy as np
from scipy import io
from time import time

import sys
sys.path.append('/home/hugh/code/TSA_Segmentation/pc2convex')
from utils import read_data, get_name, ostu3d, get_points, pc2img


def mat2npy(fpath):
    tic = time()
    mat = io.loadmat(fpath)
    fname = get_name(fpath)
    pts = np.array(mat['pts'], dtype=np.int)
    intensity = np.array(mat['intensity'])
    out = np.zeros([512, 512, 660])
    for p, i in zip(pts, intensity):
        out[tuple(p)] = i
    print(fname, 'read in in {}s'.format(time() - tic))
    return out

def npy2mat(fpath, pts, intensity):
    tic = time()
    mat = {'pts': pts, 'intensity': intensity}
    io.savemat(fpath, mat)
    
def a3d2mat(fpath, output_dir):
    data = read_data(fpath)
    fname = get_name(fpath)

    tic = time()
    # save the data as a list of points
    bita, _ = ostu3d(data, min_th=0.1, aug=1.2) # binary data
    
    pts = get_points(bita)
    intensity = []
    for pt in pts:
        intensity.append(data[tuple(pt.astype(np.int))])
    intensity = np.array(intensity).reshape([-1, 1])
    npy2mat(os.path.join(output_dir, fname + '.mat'), pts, intensity)
    print(fname, 'converted to .mat in {}s'.format(time() - tic))

fpath = '/home/hugh/code/TSA_Segmentation/pc2convex/a3d/a3d/9657d70069ba334ec5e7dad5aa189aea.a3d'
a3d2mat(fpath, './')
out = mat2npy('9657d70069ba334ec5e7dad5aa189aea.mat')
