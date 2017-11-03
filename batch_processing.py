import multiprocessing as mp
from multiprocessing import Process
from npy2mat import a3d2mat

if __name__ == '__main__':
    data_dir = glob('')
    output = './output'
    if not os.path.exists(output):
        os.mkdir(output)
    
    pool = mp.Pool()
    for f in data_dir:
        pool.apply_async(a3d2mat, args=(f, output))

    pool.close()
    pool.join()
    print('task complete')
