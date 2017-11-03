import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d
# from scipy import optimize

import config
from my_utils import get_points

def segmentation(frame, flags, bita):
    '''
    return a labeling function
    '''
    ll, rl, bd, la, hd, ra = limb_segment(frame, flags)
    # get knee
    flags['knee'] = ll[len(ll)//2, 2]
    flags['knee_upper'] = flags['knee'] + 16 // config.ratio
    flags['knee_lower'] = flags['knee'] - 16 // config.ratio
    flags['feet'] = flags['knee_lower'] // 2

    # get waist
    # use half height as proposed by TA
    flags['waist'] = flags['height'] // 2

    # belly and chest
    flags['chest'] = flags['top'] - flags['chin']
    flags['chest_th'], flags['belly_th'] = get_body_th(bita, bd, flags['chest'])

    # shoulder
    flags['left_shoulder_lower'] = bd[-1, 0] - bd[-1, 3]
    flags['right_shoulder_lower'] = bd[-1, 0] + bd[-1, 3]

    flags['left_shoulder_upper'] = la[0, 0] - la[0, 3]
    flags['right_shoulder_upper'] = ra[0, 0] + ra[0, 3]

    ll, rl, bd, la, hd, ra = map(interpz, [ll, rl, bd, la, hd, ra])
    def label(pt):
        x, y, z = pt.astype(np.int)
        if z < flags['feet']:
            pass
        return 0
    return label


def get_body_th(bita, bd, chest_z):
    # find the principle axis of the chest
    s = []
    for i in bd[:, 2]:
        if i >= chest_z:
            pts = get_points(bita[:, :, int(i)], thresh=0)
            u_, s_, v_ = np.linalg.svd(pts.astype(np.float32))
            s.append(s_)
    s = np.mean(s, axis=0)
    ang = np.arctan2(-s[1], s[0])
    return ang, np.pi / 2 + ang


def interpz(frame):
    '''
    interpolate the frame along z-axis
    :return:
    '''
    x = frame[:, 2]
    x_new = np.arange(np.amin(x), np.amax(x))
    out = []
    for i in range(frame.shape[1]):
        y = frame[:,i]
        if x is not y:
            out.append(x_new)
        else:
            f = interp1d(x, y)
            y_new = f(x_new).astype(np.int)
            out.append(y_new)
    return np.array(out)


def limb_segment(frame, flags):
    '''
    return a primitive segment
    '''
    x, y, z, b, d = frame
    z_set, z_counts = np.unique(z, return_counts=True)
    idx = np.add.accumulate(z_counts) - 1
    # use knowledge of  human structure
    legflag = True
    bodyflag = True
    headflag = True
    ll = [] # left leg
    rl = [] # right leg
    la = [] # left arm
    ra = [] # right arm
    bd = [] # body
    hd = [] # head

    for z_, c, i in zip(z_set, z_counts, idx):
        if c == 2:
            pts = np.array([(x[i - j], y[i - j], z_, b[i - j], d[i - j]) for j in range(2)])
            pts = pts[pts[:, 0].argsort()]
            if z_ <=flags['hip']:
                ll.append(pts[0])
                rl.append(pts[1])
            else:
                la.append(pts[0])
                ra.append(pts[1])
        elif c == 1:
                bd.append((x[i], y[i], z_, b[i], d[i]))
        elif c== 3:
            pts = np.array([(x[i - j], y[i - j], z_, b[i - j], d[i - j]) for j in range(3)])
            pts = pts[pts[:, 0].argsort()]
            if z_ < flags['chin']:
                la.append(pts[0])
                bd.append(pts[1])
                ra.append(pts[2])
            else:
                la.append(pts[0])
                hd.append(pts[1])
                ra.append(pts[2])
        else:
            print('Error: wrong number of centroids')
    ll = np.array(ll)
    rl = np.array(rl)
    bd = np.array(bd)
    la = np.array(la)
    hd = np.array(hd)
    ra = np.array(ra)
    return [ll, rl, bd, la, hd, ra]


# TODO: improve
# def vertex_opt(limb, kps):
#     pts = limb[:, :3]
#     res = optimize.minimize(cost, kps, args=(pts, ), method='L-BFGS-B')
#     out = res.x.reshape([-1, 3])
#     return out[:, 2]
#
# def cost(kps, pts):
#     kp_st = np.array([pts[0, 0], pts[0, 1], pts[0, 2]]).reshape([1, -1])
#     kps = kps.reshape([-1, 3])
#     # add the first and the last point in limb to kps
#     kps = np.vstack([np.array([pts[0, 0], pts[0, 1], pts[0, 2]]).reshape([1, -1]), kps])
#     kps = np.vstack([ kps, np.array([pts[-1, 0], pts[-1, 1], pts[-1, 2]]).reshape([1, -1])])
#
#     dis = 0
#     for pt in pts:
#         for i, kp in enumerate(kps):
#             if pt[2] < kp[2]:
#                 dis += norm(np.inner(pt - kp, kp - kps[i-1])) / norm(kp - kps[i-1])
#     return dis
