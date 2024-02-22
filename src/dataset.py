
import os
import logging
import multiprocessing as mp
import numpy as np
import torch
from torch.utils import data

from . import fft
from . import mrc
from . import starfile
from . import utils

logger = logging.getLogger(__name__)


def load_particles(mrcs_txt_star, lazy=False, datadir=None, relion31=False):
    """
    Load particle stack from either a .mrcs file, a .star file, a .txt file containing paths to .mrcs files,
    or a cryosparc particles.cs file

    lazy (bool): Return numpy array if True, or return list of LazyImages
    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    """
    if mrcs_txt_star.endswith('.txt'):
        particles = mrc.parse_mrc_list(mrcs_txt_star, lazy=lazy)
    elif mrcs_txt_star.endswith('.star'):
        # not exactly sure what the default behavior should be for the data paths if parsing a starfile
        try:
            particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir,
                                                                                               lazy=lazy)
        except Exception as e:
            if datadir is None:
                datadir = os.path.dirname(mrcs_txt_star)  # assume .mrcs files are in the same director as the starfile
                particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir,
                                                                                                   lazy=lazy)
            else:
                raise RuntimeError(e)
    elif mrcs_txt_star.endswith('.cs'):
        particles = starfile.csparc_get_particles(mrcs_txt_star, datadir, lazy)
    else:
        particles, _ = mrc.parse_mrc(mrcs_txt_star, lazy=lazy)
    return particles


def window_mask(resolution, in_rad, out_rad):
    assert resolution % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, resolution, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, resolution, endpoint=False, dtype=np.float32))
    r = (x0 ** 2 + x1 ** 2) ** .5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r - in_rad) / (out_rad - in_rad)))
    return mask


def downsample(imgs, resolution_out, max_threads=1):
    """
    imgs: [..., resolution, resolution]
    resolution_out: int

    output: [..., resolution_out, resolution_out]
    """
    resolution = imgs.shape[-1]
    if resolution <= resolution_out:
        return imgs
    else:
        start = int(resolution / 2 - resolution_out / 2)
        stop = int(resolution / 2 + resolution_out / 2)
        with mp.Pool(min(max_threads, mp.cpu_count())) as p:
            oldft = np.asarray(p.map(fft.ht2_center, imgs))
            newft = oldft[..., start:stop, start:stop]
            new = np.asarray(p.map(fft.iht2_center, newft))
        return new


class MRCData(data.Dataset):
    """
    Class representing an .mrcs stack file
    """

    def __init__(self, mrcfile, norm=None, invert_data=False, ind=None, window=True, datadir=None,
                 relion31=False, max_threads=16, window_r=0.85, flog=None, lazy=False, poses_gt_pkl=None,
                 resolution_input=None, no_trans=False):
        self.lazy = lazy

        if lazy or ind is not None:
            particles_real = load_particles(mrcfile, lazy=True, datadir=datadir, relion31=relion31)
            if not lazy:
                particles_real = np.array([particles_real[i].get() for i in ind])
        else:
            particles_real = load_particles(mrcfile, lazy=False, datadir=datadir, relion31=relion31)

        if not lazy:
            n_particles, ny, nx = particles_real.shape
            assert ny == nx, "Images must be square"
            assert ny % 2 == 0, "Image size must be even"
            logger.info(f"Loaded {n_particles} {ny}x{nx} images")

            # Real space window
            if window:
                logger.info(f"Windowing images with radius {window_r}")
                particles_real *= window_mask(ny, window_r, .99)

            # compute HT
            logger.info("Computing FFT")
            max_threads = min(max_threads, mp.cpu_count())

            if max_threads > 1:
                logger.info(f"Spawning {max_threads} processes")
                with mp.Pool(max_threads) as p:
                    particles = np.asarray(p.map(
                        fft.ht2_center, particles_real), dtype=np.float32)

            else:
                particles = []
                for i, img in enumerate(particles_real):
                    if i % 10000 == 0:
                        logger.info(f"{i} FFT computed")
                    particles.append(fft.ht2_center(img))

                particles = np.asarray(particles, dtype=np.float32)
                logger.info("Converted to FFT")

            if invert_data:
                particles *= -1

            # symmetrize HT
            logger.info("Symmetrizing image data")
            particles = fft.symmetrize_ht(particles)

            # normalize
            if norm is None:
                norm = [np.mean(particles), np.std(particles)]
                norm[0] = 0
            particles = (particles - norm[0]) / norm[1]
            logger.info(f"Normalized HT by {norm[0]} +/- {norm[1]}")

            self.particles = particles
            self.N = n_particles
            self.D = particles.shape[1]  # ny + 1 after symmetrizing HT
            self.norm = norm

            imgs = particles_real.astype(np.float32)
            norm_real = [np.mean(imgs), np.std(imgs)]
            imgs = (imgs - norm_real[0]) / norm_real[1]
            logger.info("Normalized real space images by "
                        f"{norm_real[0]} +/- {norm_real[1]}")

            if resolution_input is not None:
                imgs = downsample(imgs, resolution_input, max_threads=max_threads)
                logger.info("Images downsampled to "
                            f"{resolution_input}x{resolution_input}")

            self.imgs = imgs.astype(np.float32)

        else:
            self.particles_real = particles_real
            self.ind = ind

            particles_real_sample = np.array([particles_real[i].get() for i in range(1000)])
            n_particles, ny, nx = particles_real_sample.shape
            assert ny == nx, "Images must be square"
            assert ny % 2 == 0, "Image size must be even"
            self.resolution_input = resolution_input

            logger.info("Lazy loaded {len(particles_real)} {ny}x{nx} images")

            self.window = window
            self.window_r = window_r
            if window:
                logger.info(f"Windowing images with radius {window_r}")
                particles_real_sample *= window_mask(ny, window_r, .99)

            max_threads = min(max_threads, mp.cpu_count())
            logger.info(f"Spawning {max_threads} processes")
            with mp.Pool(max_threads) as p:
                particles_sample = np.asarray(p.map(
                    fft.ht2_center, particles_real_sample), dtype=np.float32)

            self.invert_data = invert_data
            if invert_data:
                particles_sample *= -1

            particles_sample = fft.symmetrize_ht(particles_sample)

            if norm is None:
                norm = [np.mean(particles_sample), np.std(particles_sample)]
                norm[0] = 0

            self.norm = norm
            self.D = particles_sample.shape[1]
            self.N = len(particles_real) if ind is None else len(ind)

            norm_real = [np.mean(particles_real_sample), np.std(particles_real_sample)]
            self.norm_real = norm_real

        self.poses_gt = None
        if poses_gt_pkl is not None:
            poses_gt = utils.load_pkl(poses_gt_pkl)
            if ind is not None:
                if poses_gt[0].ndim == 3:
                    self.poses_gt = (
                        torch.tensor(poses_gt[0][np.array(ind)]).float(),
                        torch.tensor(poses_gt[1][np.array(ind)]).float() * self.D
                    )
                else:
                    self.poses_gt = torch.tensor(poses_gt[np.array(ind)]).float()
            else:
                if poses_gt[0].ndim == 3:
                    self.poses_gt = (
                        torch.tensor(poses_gt[0]).float(),
                        torch.tensor(poses_gt[1]).float() * self.D
                    )
                else:
                    self.poses_gt = torch.tensor(poses_gt).float()

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if self.lazy:
            if self.ind is not None:
                particle_real = self.particles_real[self.ind[index]].get().astype(np.float32)
            else:
                particle_real = self.particles_real[index].get().astype(np.float32)

            if self.window:
                particle_real *= window_mask(particle_real.shape[-1], self.window_r, .99)

            particle = fft.ht2_center(particle_real)

            if self.invert_data:
                particle *= -1

            particle = fft.symmetrize_ht(particle)

            particle = (particle - self.norm[0]) / self.norm[1]

            in_dict = {'y': particle.astype(np.float32),
                       'index': index}
            particle_real = (particle_real - self.norm_real[0]) / self.norm_real[1]
            if self.resolution_input is not None:
                particle_real = downsample(particle_real, self.resolution_input)
            in_dict['y_real'] = particle_real.astype(np.float32)
        else:
            in_dict = {'y': self.particles[index],
                       'y_real': self.imgs[index],
                       'index': index}

        if self.poses_gt is not None:
            if self.poses_gt[0].ndim == 3:
                rotmat_gt = self.poses_gt[0][index]
                trans_gt = self.poses_gt[1][index]
                in_dict['R'] = rotmat_gt
                in_dict['t'] = trans_gt
            else:
                rotmat_gt = self.poses_gt[index]
                in_dict['R'] = rotmat_gt

        return in_dict

    def get(self, index):
        return self.particles[index]


class ImagePoseDataset(data.Dataset):
    def __init__(self, mrcdata, indices, predicted_rot, predicted_trans):
        """
        mrcdata: MRCData
        indices: [n_imgs]
        predicted_rot: [n_imgs, 3, 3]
        predicted_trans: [n_imgs, 2] or None
        """
        self.mrcdata = mrcdata
        self.indices = indices
        self.predicted_rot = predicted_rot
        self.predicted_trans = predicted_trans

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        mrcdata_in_dict = self.mrcdata[self.indices[index]]
        in_dict = {
            'y': mrcdata_in_dict['y'],
            'y_real': mrcdata_in_dict['y_real'],
            'index': mrcdata_in_dict['index'],
            'R': torch.tensor(self.predicted_rot[index]).float()
        }
        if self.predicted_trans is not None:
            in_dict['t'] = torch.tensor(self.predicted_trans[index]).float()
        assert mrcdata_in_dict['index'] == self.indices[index]
        return in_dict
