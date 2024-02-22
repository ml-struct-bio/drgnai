"""
Tools for dealing with SO(3) group and algebra
Adapted from https://github.com/pimdh/lie-vae
All functions are pytorch-ified
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def s2s2_to_rotmat(s2s2):
    """
    Normalize 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix.

    s2s2: [..., 6]

    output: [..., 3, 3]
    """
    v2 = s2s2[..., 3:]
    v1 = s2s2[..., 0:3]
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    e3 = torch.linalg.cross(e1, e2)
    return torch.cat([e1[..., None, :], e2[..., None, :], e3[..., None, :]], -2)


def rotmat_to_s2s2(rotmat):
    """
    rotmat: [..., 3, 3]

    output: [..., 6]
    """
    return torch.cat([rotmat[..., 0, :], rotmat[..., 1, :]], -1)


def r3_to_rotmat(r3):
    """
    Converts a view direction into a rotation matrix with a meaningless in-plane angle.
    r3: [batch_size, 3]

    output: [batch_size, 3, 3]
    """
    batch_size = r3.shape[0]
    z = r3 / r3.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    y_world = torch.tensor([0., 1., 0.]).float().to(z.device)[None].repeat(
        batch_size, 1)

    x = torch.linalg.cross(y_world, z)
    x = x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    y = torch.linalg.cross(z, x)
    y = y / x.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)

    return torch.cat([x[..., None, :], y[..., None, :], z[..., None, :]], -2)


def rotmat_to_r3(rotmat):
    """
    rotmat: [..., 3, 3]

    output: [..., 3]
    """
    return rotmat[..., 2, :]


def quat_to_rotmat(q):
    """
    Normalizes q and maps to group matrix.
    q: [..., 4]

    output: [..., 3, 3]
    """
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack(
        [
            r * r - i * i - j * j + k * k,
            2 * (r * i + j * k),
            2 * (r * j - i * k),
            2 * (r * i - j * k),
            -r * r + i * i - j * j + k * k,
            2 * (i * j + r * k),
            2 * (r * j + i * k),
            2 * (i * j - r * k),
            -r * r - i * i + j * j + k * k,
        ],
        -1,
    ).view(*q.shape[:-1], 3, 3)


def random_quat(n, dtype=torch.float32, device=None):
    u1, u2, u3 = torch.rand(3, n, dtype=dtype, device=device)
    return torch.stack((
        torch.sqrt(1-u1) * torch.sin(2 * np.pi * u2),
        torch.sqrt(1-u1) * torch.cos(2 * np.pi * u2),
        torch.sqrt(u1) * torch.sin(2 * np.pi * u3),
        torch.sqrt(u1) * torch.cos(2 * np.pi * u3),
    ), 1)


def random_rotmat(n, dtype=torch.float32, device=None):
    return quat_to_rotmat(random_quat(n, dtype, device))


def rotmat_to_euler(rotmat):
    """
    rotmat: [..., 3, 3] (numpy)

    output: [..., 3]
    """
    return Rotation.from_matrix(rotmat.swapaxes(-2, -1)).as_euler('zxz')


def euler_to_rotmat(euler):
    """
    euler: [..., 3] (numpy)

    output: [..., 3, 3]
    """
    return Rotation.from_euler('zxz', euler).as_matrix().swapaxes(-2, -1)


def symmetric_rot(rots, planes):
    """
    rots: [batch_size, 3, 3]
    plane: [batch_size / 1, 3]

    output: [batch_size, 3, 3]
    """
    in_plane_flip = torch.tensor(
        [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
        ).float().reshape(3, 3).to(rots.device)

    sym_rots_l = in_plane_flip @ rots  # in-plane rotation
    u_z = planes / planes.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    n_planes = planes.shape[0]
    e_y = torch.linalg.cross(
        u_z, torch.tensor([1.0, 0.0, 0.0]).float().expand(n_planes, 3).to(
            u_z.device), dim=-1
        )

    u_y = e_y / e_y.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    e_x = torch.linalg.cross(u_y, u_z, dim=-1)
    u_x = e_x / e_x.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    plane_basis = torch.cat([
        u_x[..., None, :], u_y[..., None, :], u_z[..., None, :]], -2)

    rotmat_r = in_plane_flip @ plane_basis
    rotmat = torch.swapaxes(plane_basis, -2, -1) @ rotmat_r
    sym_rots = sym_rots_l @ rotmat

    return sym_rots


def direction_to_azimuth_elevation(out_of_planes):
    """
    out_of_planes: [..., 3]

    up: Y
    plane: (Z, X)

    output: ([...], [...]) (azimuth, elevation)
    """
    elevation = np.arcsin(out_of_planes[..., 1])
    azimuth = np.arctan2(out_of_planes[..., 0], out_of_planes[..., 2])
    return azimuth, elevation
