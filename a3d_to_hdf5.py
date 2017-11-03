import sys
sys.path.append('/home/hugh/code/TSA_Segmentation/pc2convex')
import os
import numpy as np
import h5py
from glob import glob
from utils import read_data, get_name, ostu3d, get_points

def h5npy(fpath):
    with h5py.File(fpath, 'r') as f:
        data = f.get('data/data')
        return np.array(data)

def npyh5(fpath, array):
    with h5py.File(fpath, 'w') as f:
        f.create_dataset('data/data', array)

if __name__ == '__main__':
    data_dir = glob('/home/hugh/code/TSA_Segmentation/pc2convex/a3d/a3d/*a3d')
    output = './output/'
    if not os.path.exists(output):
        os.mkdir(output)

    for f in data_dir:
        data = read_data(f)
        fname = get_name(f)

        # save the data as a list of points
        bita, _ = ostu3d(data, min_th=0.08, aug=1.2) # binary data
        pts = get_points(bita)
        intensity = np.array([]).reshape([-1, 1])
        for pt in pts:
            np.append(intensity, data[tuple(pt)])
        out = np.hstack([pts, intensity])
        exit()
        h5f = h5py.File(output + fname + '.h5', 'w')
        h5f.create_dataset('data', data=data)
        h5f.close()
        np.savez(output + fname + '.npz')