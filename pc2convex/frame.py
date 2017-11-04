##################################################
#                                                #
# The module extract frame point from 3d tensor  #
#                                                #
##################################################

import numpy as np
from time import time
import matplotlib.pyplot as plt

import config
from my_utils import pc2img, get_height, get_points
from convex import convex_gen



def frame_gen(bita, layers, if_plot=False):
    x = []
    y = []
    z = []
    b = []
    d = []
    n_clusters = []
    height = get_height(bita)
    flags = {'hip':0,
             'shoulder':0,
             'top':0,
             'counts':0,
             'height':height,
             'height_in_layer': 0,
             'last_n_cluster':2,
             'chin':0,
             'disp': np.mean(np.diff(layers)),
             'feet': 0,
             'waist_left':0,
             'waist_right':0,
             'chest':0,
             'elbow_left': 0,
             'elbow_right': 0,
             'thigh_left': 0,
             'thigh_right': 0
             }
    tic = time()
    geom = []
    for j, i in enumerate(layers.astype(np.int)):
        if i > flags['height']:
            continue
        hypo = hypo_gen(i, height, flags['last_n_cluster'])
        pts = get_points(bita[: , :, i])
        geom, prob = convex_gen(pts)
        hypo = hypo_correction(pts, prob, geom, hypo)
        # hypo correction
        centroids, sizes = cluster_parser(geom, prob, hypo, flags)
        config.last_n_cluster = len(centroids)
        config.last_geom = geom

        if config.debug:
            print(i, 'raw data: ',len(centroids), hypo, prob)

        # number of cluster stays the same for a few layers before it changes
        if flags['last_n_cluster'] == len(centroids):
            flags['counts'] += 1
        else:
            flags['counts'] = 1

        # when n cluster changes, update the flags with a rough approximation
        flags['height_in_layer'] += 1
        if flags['last_n_cluster']==2 and len(centroids)==1:
            flags['hip'] = i - flags['disp'] / 2
        elif flags['last_n_cluster']==1 and len(centroids)==3:
            flags['shoulder'] = i
            flags['shoulder_in_layer'] = j
        elif flags['last_n_cluster']==3 and len(centroids)==2:
            flags['top'] = i

        flags['last_n_cluster'] = len(centroids)
        for cen, sz in zip(centroids, sizes):
            x.append(cen[0])
            y.append(cen[1])
            z.append(i)
            n_clusters.append(len(centroids))
            b.append(sz[0])
            d.append(sz[1])

    flags['chin'] = flags['top'] - (flags['top'] - flags['shoulder']) * (3/4)
    toc = time()
    print('frame generated in {} s'.format(toc - tic))

    if if_plot:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        ax1, ax2 = axes
        img1 = pc2img(bita, axis=1)
        img2 = pc2img(bita, axis=0)
        rgb_img1 = np.swapaxes(np.swapaxes(np.stack((img1,) * 3), 0, 2), 0, 1)
        rgb_img2 = np.swapaxes(np.swapaxes(np.stack((img2,) * 3), 0, 2), 0, 1)
        ax1.imshow(np.flipud(rgb_img1), origin='lower')
        ax2.imshow(np.flipud(rgb_img2), origin='lower')
        for i in range(len(x)):
            ax1.plot(x[i], z[i], 'o', linewidth=5, color='r')
            ax1.plot(x[i] + b[i], z[i], 'o', linewidth=1, color='g')
            ax1.plot(x[i] - b[i], z[i], 'o', linewidth=1, color='g')

            ax2.plot(y[i], z[i], 'o', linewidth=5, color='r')
            ax2.plot(y[i] + d[i], z[i], 'o', linewidth=1, color='g')
            ax2.plot(y[i] - d[i], z[i], 'o', linewidth=1, color='g')
        plt.show()

    return  (x, y, z, b, d), flags


def hypo_gen(layer, height, last_n_clusters):
    if layer < (height / 2):
        if last_n_clusters == 2 and layer < height / 2:
            return [1, 2]# [3]
        else:
            return [1]#, [2,3]
    elif layer < height / 3 * 2:
            return [1]
    else:
        if last_n_clusters == 1:
            return [1, 3]#, [2]
        if last_n_clusters == 3:
            return [2, 3]#, [1]
        else:
            return [2] #, [1,3]


def hypo_correction(pts, prob, geom, hypo):
    center = geom[1][4] + 256 / config.ratio
    # case1: the two cluster are close
    if geom[1][0] <= geom[1][1] * 1.1:
        if 1 in hypo and 2 in hypo and prob[1] < prob[0]:
            if np.ptp(pts[np.argsort(np.abs(pts[:, 0] - center))[:len(pts) // 6]][:, 1]) < np.ptp(pts[:,1]) * config.weak_link:
                hypo.remove(1)
                if config.debug:
                    print('[hypo_correction]: There is a weak connection in the center thus 1 is removed')
        elif 1 in hypo and 2 in hypo and prob[1] > prob[0] :
            if np.ptp(pts[np.argsort(np.abs(pts[:, 0] - center))[:len(pts) // 6]][:, 1]) > np.ptp(pts[:,1]) * config.strong_link:
                hypo.remove(2)
                if config.debug:
                    print('[hypo_correction: There is a strong connection in the center thus 2 is removed')

    # case2: the side cluster is minor
    if 2 not in hypo and 3 in hypo and prob[1] > prob[2] and prob[2] > prob[0]:
        hypo.remove(3)
        if config.debug:
            print('[hypo_correction: side cluster is trivial thus 1 is appended to hypo')

    # case3: unbalanced cluster
    if 1 in hypo and 3 in hypo and prob[2] > prob[0]:
        if np.amax(pts[:, 0]) < geom[2][0] + geom[2][1] * 1.2 + geom[2][3]:
            hypo.remove(3)
            if config.debug:
                print('unbalanced side cluster')
    return hypo


def cluster_parser(x, probs, hypo, flags):
    probs_ = [p if i+1 in hypo else 0 for i, p in enumerate(probs)]
    n_cluster = np.argmax(probs_) + 1
    if flags['counts'] < config.min_count:
        if config.debug:
            print('count < than min_count')
        n_cluster = flags['last_n_cluster']
    if n_cluster == 1:
        return [(x[0][0] + 256//config.ratio, x[0][2] + 256//config.ratio)], [(x[0][1], x[0][3])]
    elif n_cluster == 2:
        return [(x[1][0] + 256//config.ratio + x[1][4], x[1][2] + 256//config.ratio), (-x[1][0] + 256//config.ratio + x[1][4], x[1][2] + 256//config.ratio)], [(x[1][1], x[1][3]), (x[1][1], x[1][3])]
    elif n_cluster == 3:
        return [
            [(x[2][0] + x[2][1] + x[2][3] + 256//config.ratio + x[2][5], x[2][2] + 256//config.ratio),
             (256//config.ratio + x[2][5], x[2][2] + 256//config.ratio),

             (-x[2][0] - x[2][1] - x[2][3] + 256//config.ratio + x[2][5], x[2][4] + 256//config.ratio)],
            [(x[2][1], x[2][1]), (x[2][3], x[2][3]), (x[2][1], x[2][1])]
        ]

