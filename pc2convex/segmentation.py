import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d
# from scipy import optimize
from time import time
import config
from my_utils import get_points, pc2img


def label_pts(pts, label_fcn, scaled=True):
    tic = time()
    labels = []
    pts = np.array(pts)
    if not scaled:
        pts = (pts / config.ratio).astype(np.int)
    for pt in pts:
        labels.append(label_fcn(pt))
    print('labeling finished in {}s'.format(time() - tic))
    return np.array(labels).ravel()


def segmentation(frame, flags, bita):
    '''
    return a labeling function
    '''
    rl, ll, bd, ra, hd, la = limb_segment(frame, flags)
    # get knee
    flags['knee'] = ll[len(ll)//2, 2]
    #flags['knee_upper'] = flags['knee'] + 16 // config.ratio
    #flags['knee_lower'] = flags['knee'] - 16 // config.ratio
    
    flags['knee_lower'] = flags['knee'] + 16 // config.ratio
    flags['knee_upper'] = (flags['knee_lower'] + ll[-1, 2]) / 2
    flags['feet'] = flags['knee_lower'] // 2
    # get waist
    # use half height as proposed by TA

    flags['thigh_left']  = ll[-1, 0] + ll[-1, 3] / 3
    flags['thigh_right'] = rl[-1, 0] - rl[-1, 3] / 3
    # flags['waist'] = bd[flags['height_in_layer'] // 2 - len(ll), 0] + 60 // config.ratio

    flags['waist_right']  = np.average(bd[:, 0]) - ll[-1, 3] / 2
    flags['waist_left'] = np.average(bd[:,0]) + rl[-1, 3] / 2

    # belly and chest

    # lower shoulder
    indx = flags['shoulder_in_layer'] - len(ll) - 3
    flags['shoulder_lower'] = bd[indx, 2]
    flags['shoulder_left_lower'] = bd[indx, 0] + bd[indx, 3]
    flags['shoulder_right_lower'] = bd[indx, 0] - bd[indx, 3]

    # upper shoulder
    flags['shoulder_upper'] = hd[1,2]
    flags['shoulder_left_upper'] = hd[1, 0] + hd[1, 3]
    flags['shoulder_right_upper'] = hd[1, 0] - hd[1, 3]

    #elbow
    elbow_indx1 = sum(np.diff((la[:len(la)//3*2, 0])) >= 0)
    elbow_indx2 = sum(np.diff((ra[:len(ra)//3*2, 0])) < 0)
    flags['elbow'] = (la[elbow_indx1, 2] + ra[elbow_indx2, 2]) / 2.0


    # interpolate the centroids and the bounds
    ll, rl, bd, la, hd, ra = map(interpz, [ll, rl, bd, la, hd, ra])
    

    # ---------- CORRECTION ---------- #
    #TODO: remove all correction
    #neck corretion
    # hd_b = hd[2, 0]
    # hd_m = hd[2, len(hd)//2]
    # hd_r = int(np.average(hd[1,:])-np.amax(hd[4,:]))
    # hd_l = int(np.average(hd[1,:]))
    
    # tmp = np.argmin(np.average(pc2img(bita, axis=0)[hd_b:hd_m, hd_r:hd_l], axis=1)) + hd[2, 0]
    #print('chin corrected from {} to {}'.format(flags['chin'], tmp))
    flags['chin'] = flags['top'] - max(flags['top'] // 8, 8)
    flags['chest'] = int(flags['chin'] + (flags['chin'] - flags['top']) * config.chest2head)
    #print(flags['chest'])
    flags['chest_th'], flags['belly_th'] = get_body_th(bita, bd.T, flags['chest'])

    flags['waist'] = flags['hip'] + (flags['chest'] - flags['hip']) / 3

    # ---------- CORRECTION ---------- #

    def label(pt):
        x, y, z = pt
        if z < flags['feet']:
            if inconvex(pt, ll, relax_factor=config.feet_relax):
                return 16
            elif inconvex(pt, rl, relax_factor=config.feet_relax):
                return 15
            else:
                return -1

        elif z < flags['knee_lower']:
            if inconvex(pt, ll, relax_factor=config.leg_relax):
                return 14
            elif inconvex(pt, rl, relax_factor=config.leg_relax):
                return 13
            else:
                return -1

        elif z < flags['knee_upper']:
            if inconvex(pt, ll, relax_factor=config.knee_relax):
                return 12
            elif inconvex(pt, rl, relax_factor=config.knee_relax):
                return 11
            else:
                return -1
            
        elif z <= flags['hip']:
            if inconvex(pt, ll, relax_factor=config.thigh_relax) or inconvex(pt, rl, relax_factor=config.thigh_relax) or inconvex(pt, bd, relax_factor=config.thigh_relax):
                indx = where1d(ll[2, :], z)
                if x > ll[0, indx] - ll[3, indx] / 3:
                    return 10
                elif x < rl[0, indx] + rl[3, indx] / 3:
                    return 8
                else:
                    return 9
            else:
                return -1

        elif z <= flags['waist']:
            if (inconvex(pt, ll, relax_factor=config.thigh_relax) or
                inconvex(pt, rl, relax_factor=config.thigh_relax) or
                inconvex(pt, bd, relax_factor=config.thigh_relax)):
                # TODO: reverse ll and rl now. Need to fix!!!!
                if below((x, z),
                         [(ll[0, -1] - ll[3, -1]//6, ll[2, -1]), (flags['waist_left'], flags['waist'])]):
                    return 10
                elif below((x, z),
                           [(rl[0, -1] + rl[3, -1]//6, rl[2, -1]), (flags['waist_right'], flags['waist'])]):
                    return 9
                else:
                    return 8
            else:
                return -1
        elif z<= flags['chest']:
            if inconvex(pt, bd, relax_factor=config.chest_relax):
                indx = where1d(bd[2, :], z)
                if below((x, y), [(bd[0, indx], bd[1, indx]), (bd[0, indx] + np.cos(flags['belly_th']), bd[1, indx] + np.sin(flags['belly_th']))]):
                    return 7
                else:
                    return 6
            else:
                return -1
        elif z <= flags['chin']:
            if below((x,z), [(flags['shoulder_left_lower'], flags['shoulder_lower']),
                             (flags['shoulder_left_upper'], flags['shoulder_upper'])]):
                return 3
            elif above((x,z), [(flags['shoulder_right_lower'], flags['shoulder_lower']),
                             (flags['shoulder_right_upper'], flags['shoulder_upper'])]):
                return 1
            elif inbox(pt, frame=bd, bound=[la[0, 0] - bd[0, 0], 0]):
                indx = where1d(bd[2, :], z)
                bd_x = bd[0, indx]
                bd_y = bd[1, indx]
                bd_d = bd[4, indx]
                if below((x, y), [(bd_x, bd_y+bd_d/5),
                                  (bd_x + np.cos(flags['chest_th']),
                                   bd_y + np.sin(flags['chest_th'])+bd_d/5)]):
                    return 5
                else:
                    return 17
            else:
                return -1

        else:
            if inconvex(pt, hd, relax_factor=config.head_relax):
                if z <= flags['top']:
                    return 0
            if inconvex(pt, la, relax_factor=config.arm_relax):
                if z <= flags['elbow']:
                    return 3
                else:
                    return 4
            elif inconvex(pt, ra, relax_factor=config.arm_relax):
                if z <= flags['elbow']:                
                    return 1
                else:
                    return 2
            else:
                return -1
        # else:
        #     if inconvex(pt, la, relax_factor=config.arm_relax):
        #         return 3
        #     elif inconvex(pt, ra, relax_factor=config.arm_relax):
        #         return 1
        #     else:
        #         return -1

    return label

#test pass
def above(pt, kps):
    '''
    return True if pt in above the vector(kps[1] - kps[0])
    '''

    assert len(kps) == 2
    assert len(pt) == 2
    pt = np.array(pt)
    kps = np.array(kps)
    if np.cross(kps[1] - kps[0], pt - kps[0]) >= 0:
        return True
    else:
        return False

#test pass
def below(pt, kps):
    '''
    return True if pt in above the vector(kps[1] - kps[0])
    '''

    assert len(kps) == 2
    assert len(pt) == 2
    pt = np.array(pt)
    kps = np.array(kps)
    if np.cross(kps[1] - kps[0], pt - kps[0]) < 0:
        return True
    else:
        return False

def where1d(array, x):
    if x <= np.amin(array):
        return 0
    elif x >= np.amax(array):
        return -1
    else:
        return np.where(array == x)[0][0]

def inconvex(pt, frame, relax_factor=0):
    px, py, pz = np.array(pt)
    frame = np.array(frame)
    indx = where1d(frame[2, :], pt[2])
    fx, fy, fz, fb, fd = frame[:, indx]
    if (px - fx)**2 / fb**2 + (py - fy)**2 / fd**2 < (1 + relax_factor):
        return True
    else:
        return False

def inbox(pt, frame, bound=list([0, 0])):
    x, y, z = pt
    indx = where1d(frame[2, :], pt[2])

    fx, fy, fz, fb, fd = frame[:, indx]
    xb, yb = bound
    xb = max(xb, fb)
    yb = max(yb, fd)
    if fx - xb <= x and x <= fx + xb:
        if fy - yb <= y and y <= fy + yb:
            return True

    return False


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
        y = frame[:, i]
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
