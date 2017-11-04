import sys
import os
from time import time
from glob import glob
from scipy import io
import multiprocessing as mp
from multiprocessing import Process, Pool

sys.path.append('../pc2convex')
sys.path.append('../npy2mat')

from npy2mat import mat2npy
from my_utils import *
from frame import frame_gen
from segmentation import label_pts, segmentation
import config

if_vis = True
vis_dir = '~/data/seg_vis/'
data_path = '~/data/a3d/*a3d'

def label_mat(fpath):
    tic = time()
    fname = get_name(fpath)
    dset = io.readmat(fpath)
    data = mat2npy(dset)
    bita = compress_data(data>0, config.ratio)
    n_layers = 30
    layers = np.linspace(0, bita.shape[2] - 1, n_layers)
    frames, flags = frame_gen(bita, layers, if_plot=False)
    label_fcn = segmentation(frames, flags, bita)
    dset['labels'] = label_pts(dset['pts'], label_fcn=label_fcn)
    io.savemat(fpath, mat)
    toc = time()
    print(fname, ' is labeled in {}s'.format(toc - tic))
    if if_vis:
        pts = get_points(bita, thresh=0)
        seg_vis2d(dset['pts'], dset['labels'], fname, output_dir=vis_dir, savefig=True)
        print(fname, 'is visualized in {}s'.format(time() - toc))


if __name__ == '__main__':
    data_dir = glob(data_path)

    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    pool = Pool()
    for f in data_dir:
        pool.apply_async(label_mat, args=(f,))

    pool.close()
    pool.join()
    print('task complete')

