import sys
import os
from time import time
from glob import glob
from scipy import io
import multiprocessing as mp

sys.path.append('../pc2convex')
sys.path.append('../npy2mat')

from npy2mat import mat2npy
from my_utils import *
from frame import frame_gen
from segmentation import label_pts, segmentation
import config

if_vis = True
# on server
#vis_dir = '../home/seg/data/seg_vis/'
#data_path = '/home/seg/data/a3d2npy/*'

#on laptop
vis_dir = '../output/seg_vis/'
data_path = '../data/*'

def label_mat(fpath, vis_dir):
    tic = time()
    fname = get_name(fpath)
    dset = io.loadmat(fpath)
    data = mat2npy(dset)
    bita = compress_data(data>0, config.ratio)
    n_layers = 30
    layers = np.linspace(0, bita.shape[2] - 1, n_layers)
    frames, flags = frame_gen(bita, layers, if_plot=False)
    label_fcn = segmentation(frames, flags, bita)
    print(flags)
    # labels = label_pts(dset['pts'], label_fcn=label_fcn, scaled=False)
    # dset['labels'] = labels
    # io.savemat(fpath, dset)
    toc = time()
    print(fname, ' is labeled in {}s'.format(toc - tic))
    if if_vis:
        print('visualization is enabled')
        pts_ = get_points(bita, thresh=0)
        labels_ = label_pts(pts_, label_fcn, scaled=True)
        seg_vis2d(pts_, labels_, fname, output_dir=vis_dir, savefig=True)
        print(fname, 'is visualized in {}s'.format(time() - toc))


if __name__ == '__main__':
    data_dir = glob(data_path)
    print('{} .mat files in total'.format(len(data_dir)))
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    pool = mp.Pool()
    
    print('{} core is available.'.format(mp.cpu_count()))
    for f in data_dir:
        pool.apply_async(label_mat, args=(f, vis_dir))

    pool.close()
    pool.join()
    print('task complete')

