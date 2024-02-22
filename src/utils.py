
import os
import subprocess
import pickle
import logging
from pathlib import Path

import collections
import functools
import torch
import numpy as np

logger = logging.getLogger(__name__)


class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


def load_pkl(pkl):
    with open(pkl,'rb') as f:
        x = pickle.load(f)

    return x


def save_pkl(data, out_pkl, mode='wb'):
    if mode == 'wb' and os.path.exists(out_pkl):
        logger.warning(f"Warning: {out_pkl} already exists. Overwriting.")
    with open(out_pkl, mode) as f:
        pickle.dump(data, f)


def to_numpy(t):
    return t.detach().cpu().numpy()


def R_from_eman(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    # handling EMAN convention mismatch for where the origin of an image is (bottom right vs top right)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1

    return R

def R_from_relion(a,b,y):
    a *= np.pi/180.
    b *= np.pi/180.
    y *= np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)
    Ra = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Rb = np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])
    Ry = np.array(([cy,-sy,0],[sy,cy,0],[0,0,1]))
    R = np.dot(np.dot(Ry,Rb),Ra)
    R[0,1] *= -1
    R[1,0] *= -1
    R[1,2] *= -1
    R[2,1] *= -1

    return R


def R_from_relion_scipy(euler_, degrees=True):
    '''Nx3 array of RELION euler angles to rotation matrix'''
    from scipy.spatial.transform import Rotation as RR
    euler = euler_.copy()
    if euler.shape == (3,):
        euler = euler.reshape(1,3)
    euler[:,0] += 90
    euler[:,2] -= 90
    f = np.ones((3,3))
    f[0,1] = -1
    f[1,0] = -1
    f[1,2] = -1
    f[2,1] = -1
    rot = RR.from_euler('zxz', euler, degrees=degrees).as_matrix()*f

    return rot


def R_to_relion_scipy(rot, degrees=True):
    '''Nx3x3 rotation matrices to RELION euler angles'''
    from scipy.spatial.transform import Rotation as RR
    if rot.shape == (3,3):
        rot = rot.reshape(1,3,3)
    assert len(rot.shape) == 3, "Input must have dim Nx3x3"
    f = np.ones((3,3))
    f[0,1] = -1
    f[1,0] = -1
    f[1,2] = -1
    f[2,1] = -1
    euler = RR.from_matrix(rot*f).as_euler('zxz', degrees=True)
    euler[:,0] -= 90
    euler[:,2] += 90
    euler += 180
    euler %= 360
    euler -= 180
    if not degrees:
        euler *= np.pi/180
    return euler


def xrot(tilt_deg):
    '''Return rotation matrix associated with rotation over the x-axis'''
    theta = tilt_deg*np.pi/180
    tilt = np.array([[1.,0.,0.],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

    return tilt

@memoized
def _zero_sphere_helper(D):
    xx = np.linspace(-1, 1, D, endpoint=True if D % 2 == 1 else False)
    z,y,x = np.meshgrid(xx,xx,xx)
    coords = np.stack((x,y,z),-1)
    r = np.sum(coords**2,axis=-1)**.5

    return np.where(r>1)


def zero_sphere(vol):
    """Zero values of @vol outside the sphere."""

    if len(set(vol.shape)) != 1:
        raise ValueError('volume must be a cube!')

    D = vol.shape[0]
    tmp = _zero_sphere_helper(D)
    logger.info('Zeroing {} pixels'.format(len(tmp[0])))
    vol[tmp] = 0

    return vol


def make_summary(pose_pred_params, w_eps, bimodal=False, N_img=100):
    # fixme: only implemented for bimodal mode for now
    if bimodal:
        pose_params_summary = {}
        pose_params_summary['R1'] = pose_pred_params['R1'][:N_img].detach().cpu().numpy()
        pose_params_summary['R2'] = pose_pred_params['R2'][:N_img].detach().cpu().numpy()
        w_eps = w_eps.reshape(2, -1, 3)
        pose_params_summary['w_eps_0'] = w_eps[0, :N_img].detach().cpu().numpy()
        pose_params_summary['w_eps_1'] = w_eps[1, :N_img].detach().cpu().numpy()
        pose_params_summary['p'] = pose_pred_params['p'][:N_img].detach().cpu().numpy()

    else:
        pose_params_summary = {}

    return pose_params_summary


def get_w_eps_std(w_eps, pose_pred_params, bimodal=False):
    if bimodal:
        p = pose_pred_params['p']
        p_cat = torch.cat((1. - p, p), 0)
        w_eps_pred = w_eps[p_cat.reshape(-1) > 0.5]
        w_eps_std = torch.std(w_eps_pred).item()
    else:
        w_eps_std = torch.std(w_eps).item()

    return w_eps_std


def checksum(outdir: str) -> float:
    out_path = Path(outdir)
    csum = 0

    weights_fls = tuple(out_path.glob("weights.*.pkl"))
    if not weights_fls:
        raise ValueError(f"No weights outputs in directory `{outdir}` which "
                         "does not look like drgnai experiment output!")

    for weights_fl in sorted(weights_fls):
        hvolm = torch.load(weights_fl)['hypervolume_state_dict']

        csum += round(hvolm['mlp.main.0.weight'].sum().item(), 8)

    for conf_fl in sorted(out_path.glob("conf.*.pkl")):
        conf = load_pkl(conf_fl)
        zdim = conf.shape[1]

        rep_vec = np.tile([1, -1], zdim // 2 + zdim % 2)[:zdim]
        csum += round((conf @ rep_vec).sum(), 8)

    return round(csum, 8)


def run_command(cmd: str) -> tuple[str, str]:
    cmd_out = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return cmd_out.stdout, cmd_out.stderr
