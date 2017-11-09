import multiprocessing as mp
from npy2mat import a3d2mat

if __name__ == '__main__':
    data_dir = glob('/home/seg/data/P')
    output = './output'
    if not os.path.exists(output):
        os.mkdir(output)
    
    pool = mp.Pool()
    for f in data_dir:
        pool.apply_async(a3d2mat, args=(f, output))

    pool.close()
    pool.join()
    print('task complete')
