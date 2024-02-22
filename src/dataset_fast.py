
import multiprocessing as mp
import numpy as np
import logging
from typing import Tuple, Union
from multiprocessing import Pool
from collections import Counter, OrderedDict
from scipy.spatial.transform import Rotation

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

from . import fft
from . import starfile
from . import utils
from .source import ImageSource

logger = logging.getLogger(__name__)


def window_mask(resolution, in_rad, out_rad):
    assert resolution % 2 == 0
    x0, x1 = np.meshgrid(np.linspace(-1, 1, resolution, endpoint=False, dtype=np.float32),
                         np.linspace(-1, 1, resolution, endpoint=False, dtype=np.float32))
    r = (x0 ** 2 + x1 ** 2) ** .5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r - in_rad) / (out_rad - in_rad)))
    return mask


def window_mask_tensor(resolution, in_rad, out_rad):
    """
    Create a square radial mask of linearly-interpolated float values
    from 1.0 (within in_rad of center) to 0.0 (beyond out_rad of center)
    Args:
        D: Side length of the (square) mask
        in_rad: inner radius (fractional float between 0 and 1) inside which all values are 1.0
        out_rad: outer radius (fractional float between 0 and 1) beyond which all values are 0.0

    Returns:
        A 2D Tensor of shape (D, D) of mask values between 0 (inclusive) and 1 (inclusive)
    """
    assert resolution % 2 == 0
    assert in_rad <= out_rad
    x0, x1 = torch.meshgrid(
        torch.linspace(-1, 1, resolution + 1, dtype=torch.float32)[:-1],
        torch.linspace(-1, 1, resolution + 1, dtype=torch.float32)[:-1],
    )
    r = (x0**2 + x1**2) ** 0.5
    mask = torch.minimum(
        torch.tensor(1.0),
        torch.maximum(torch.tensor(0.0), 1 - (r - in_rad) / (out_rad - in_rad)),
    )
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
        with Pool(min(max_threads, mp.cpu_count())) as p:
            oldft = np.asarray(p.map(fft.ht2_center, imgs))
            newft = oldft[..., start:stop, start:stop]
            new = np.asarray(p.map(fft.iht2_center, newft))
        return new


class ImageDataset(data.Dataset):
    def __init__(
        self,
        mrcfile,
        tilt_mrcfile=None,
        lazy=True,
        norm=None,
        keepreal=False,
        invert_data=False,
        ind=None,
        window=True,
        datadir=None,
        window_r=0.85,
        max_threads=16,
        device: Union[str, torch.device] = "cpu",
        resolution_input=None,
        poses_gt_pkl=None,
        no_trans=False
    ):
        assert not keepreal, "Not implemented yet"
        datadir = datadir or ""
        self.ind = ind
        self.src = ImageSource.from_file(
            mrcfile,
            lazy=lazy,
            datadir=datadir,
            indices=ind,
            max_threads=max_threads,
        )
        if tilt_mrcfile is None:
            self.tilt_src = None
        else:
            self.tilt_src = ImageSource.from_file(
                tilt_mrcfile, lazy=lazy, datadir=datadir
            )

        ny = self.src.D
        assert ny % 2 == 0, "Image size must be even."

        self.N = self.src.n  # number of particles
        self.Nt = self.N  # number of tilts
        self.D = ny + 1  # after symmetrization
        self.invert_data = invert_data
        self.window = window_mask_tensor(ny, window_r, 0.99).to(device) if window else None
        norm = norm or self.estimate_normalization()
        self.norm = [float(x) for x in norm]
        norm_real = self.estimate_normalization_real()
        self.norm_real = [float(x) for x in norm_real]
        self.device = device
        self.lazy = lazy
        self.resolution_input = resolution_input
        self.max_threads = max_threads
        self.subtomogram_averaging = False

        self.rot_gt = None
        if poses_gt_pkl is not None:
            poses_gt = utils.load_pkl(poses_gt_pkl)
            if poses_gt[0].ndim == 3:
                self.rot_gt = poses_gt[0]
                self.trans_gt = poses_gt[1] * self.D
                self.has_trans = True
            else:
                self.rot_gt = poses_gt
                self.has_trans = False

    def estimate_normalization(self, n=1000):
        n = min(n, self.N) if n is not None else self.N
        indices = range(0, self.N, self.N // n)  # FIXME: what if the data is not IID??
        imgs = self.src.images(indices)

        particleslist = []
        for img in imgs:
            particleslist.append(fft.ht2_center(img))
        imgs = np.array(particleslist)

        if self.invert_data:
            imgs *= -1

        imgs = fft.symmetrize_ht(imgs)
        norm = (0, np.std(imgs))
        logger.info("Normalizing HT by {} +/- {}".format(*norm))

        return norm

    def estimate_normalization_real(self, n=1000):
        n = min(n, self.N) if n is not None else self.N
        indices = range(0, self.N, self.N // n)  # FIXME: what if the data is not IID??
        imgs = self.src.images(indices)
        norm = (torch.mean(imgs), torch.std(imgs))
        logger.info('Normalized real space images by {} +/- {}'.format(*norm))

        return norm

    def _process(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if self.window is not None:
            data *= self.window
        if self.invert_data:
            data *= -1
        f_data = fft.ht2_center(data, tensor=True)
        f_data = fft.symmetrize_ht_torch(f_data)
        f_data = (f_data - self.norm[0]) / self.norm[1]
        r_data = (data - self.norm_real[0]) / self.norm_real[1]
        assert self.resolution_input is None, "downsampling not implemented"
        return r_data, f_data

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if isinstance(index, list):
            index = torch.Tensor(index).to(torch.long)

        r_particles, particles = self._process(self.src.images(index).to(self.device))
        if self.tilt_src is None:
            # If no tilt data is present because a tilt_mrcfile was not specified,
            # we simply return a reference to the particle data to avoid consuming
            # any more memory while conforming to torch.Dataset's type/shape expectations,
            # and rely on the caller to properly interpret it.
            # TODO: Find a more robust way to do this.
            r_tilt = r_particles
            tilt = particles
        else:
            r_tilt, tilt = self._process(self.tilt_src.images(index).to(self.device))

        if isinstance(index, int):
            logger.debug(f"ImageDataset returning images at index ({index})")
        else:
            logger.debug(
                f"ImageDataset returning images for {len(index)} indices ({index[0]}..{index[-1]})"
            )

        return r_particles, particles, r_tilt, tilt, index

    def get_slice(self, start: int, stop: int):
        assert self.tilt_src is None
        return self.src.images(slice(start, stop), require_contiguous=True).numpy(), None


class TiltSeriesData(ImageDataset):
    """
    Class representing tilt series
    """

    def __init__(
            self,
            tiltstar,
            ntilts,
            angle_per_tilt,
            random_tilts=False,
            ind=None,
            voltage=None,
            expected_res=None,
            dose_per_tilt=None,
            tilt_axis_angle=0.,
            **kwargs
    ):
        # Note: ind is the indices of the *tilts*, not the particles
        super().__init__(tiltstar, ind=ind, **kwargs)

        # Parse unique particles from _rlnGroupName
        s = starfile.Starfile.load(tiltstar)
        if ind is not None:
            s.df = s.df.loc[ind]
        group_name = list(s.df["_rlnGroupName"])
        particles = OrderedDict()
        for ii, gn in enumerate(group_name):
            if gn not in particles:
                particles[gn] = []
            particles[gn].append(ii)
        self.particles = [np.asarray(pp, dtype=int) for pp in particles.values()]
        self.Nt = self.N  # number of tilts
        self.N = len(particles)  # number of particles
        self.ctfscalefactor = np.asarray(s.df["_rlnCtfScalefactor"], dtype=np.float32)
        self.tilt_numbers = np.zeros(self.Nt)
        for ind in self.particles:
            sort_idxs = self.ctfscalefactor[ind].argsort()
            ranks = np.empty_like(sort_idxs)
            ranks[sort_idxs[::-1]] = np.arange(len(ind))
            self.tilt_numbers[ind] = ranks
        self.tilt_numbers = torch.tensor(self.tilt_numbers).to(self.device)
        logger.info(f"Loaded {self.Nt} tilts for {self.N} particles")

        counts = Counter(group_name)
        unique_counts = set(counts.values())
        logger.info(f"{unique_counts} tilts per particle")

        self.counts = counts
        assert ntilts <= min(unique_counts)
        self.ntilts = ntilts
        self.random_tilts = random_tilts

        self.voltage = voltage
        self.dose_per_tilt = dose_per_tilt

        # Assumes dose-symmetric tilt scheme
        # As implemented in Hagen, Wan, Briggs J. Struct. Biol. 2017
        self.tilt_angles = None
        if angle_per_tilt is not None:
            self.tilt_angles = angle_per_tilt * torch.ceil(self.tilt_numbers / 2)
            self.tilt_angles = torch.tensor(self.tilt_angles).to(self.device)

        # the tilting is not perfectly done around Y
        # see pose_consistency.ipynb
        tilt_scheme = []
        for i in range(ntilts):
            tilt_scheme.append(angle_per_tilt * np.ceil(i / 2.) * (((np.floor(i / 2.) + 1) % 2) * 2. - 1.))
        tilts = [Rotation.from_euler('zyz', [tilt_axis_angle * np.pi / 180., t * np.pi / 180., 0.]) for t in tilt_scheme]
        tilt_rots = [Rotation.as_matrix(t) for t in tilts]
        self.tilt_rots = torch.tensor(tilt_rots).float()

        self.subtomogram_averaging = True

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if isinstance(index, list):
            index = torch.Tensor(index).to(torch.long)
        tilt_indices = []
        for ii in index:
            if self.random_tilts:
                tilt_index = np.random.choice(
                    self.particles[ii], self.ntilts, replace=False
                )
            else:
                # take the first ntilts
                tilt_index = self.particles[ii][0: self.ntilts]
            tilt_indices.append(tilt_index)
        tilt_indices = np.concatenate(tilt_indices)
        r_particles, particles = self._process(self.src.images(tilt_indices).to(self.device))

        in_dict = {}

        if self.rot_gt is not None:
            if self.ind is not None:
                rots = self.rot_gt[self.ind][tilt_indices]
            else:
                rots = self.rot_gt[tilt_indices]
            rots = torch.from_numpy(rots).float().reshape(-1, self.ntilts, 3, 3)

            trans = None
            if self.has_trans:
                if self.ind is not None:
                    trans = self.trans_gt[self.ind][tilt_indices]
                else:
                    trans = self.trans_gt[tilt_indices]
                trans = torch.from_numpy(trans).float().reshape(-1, self.ntilts, 2)

            in_dict['R'] = rots  # batch_size, n_tilts, 3, 3
            in_dict['t'] = trans  # batch_size, n_tilts, 2

        particles = particles.reshape(-1, self.ntilts, *particles.shape[-2:])
        tilt_indices = tilt_indices.reshape(-1)
        r_particles = r_particles.reshape(-1, self.ntilts, *r_particles.shape[-2:])

        in_dict['y'] = particles  # batch_size, n_tilts, D, D
        in_dict['index'] = index  # batch_size
        in_dict['tilt_index'] = tilt_indices  # batch_size * n_tilts
        in_dict['y_real'] = r_particles  # batch_size, n_tilts, D - 1, D - 1

        return in_dict

    def get_tilt(self, index):
        return super().__getitem__(index)

    def get_slice(self, start: int, stop: int) -> Tuple[np.ndarray, np.ndarray]:
        # we have to fetch all the tilts to stay contiguous, and then subset
        tilt_indices = [self.particles[index] for index in range(start, stop)]
        cat_tilt_indices = np.concatenate(tilt_indices)
        images = self.src.images(cat_tilt_indices, require_contiguous=True)

        tilt_masks = []
        for tilt_idx in tilt_indices:
            tilt_mask = np.zeros(len(tilt_idx), dtype=np.bool)
            if self.random_tilts:
                tilt_mask_idx = np.random.choice(
                    len(tilt_idx), self.ntilts, replace=False
                )
                tilt_mask[tilt_mask_idx] = True
            else:
                # i = (len(tilt_idx) - self.ntilts) // 2
                i = 0
                tilt_mask[i: i + self.ntilts] = True
            tilt_masks.append(tilt_mask)
        tilt_masks = np.concatenate(tilt_masks)
        selected_images = images[tilt_masks]
        selected_tilt_indices = cat_tilt_indices[tilt_masks]

        return selected_images.numpy(), selected_tilt_indices

    def critical_exposure(self, freq):
        assert self.voltage is not None, \
            "Critical exposure calculation requires voltage"

        assert self.voltage == 300 or self.voltage == 200, \
            "Critical exposure calculation requires 200kV or 300kV imaging"

        # From Grant and Grigorieff, 2015
        scale_factor = 1
        if self.voltage == 200:
            scale_factor = 0.75
        critical_exp = torch.pow(freq, -1.665)
        critical_exp = torch.mul(critical_exp, scale_factor * 0.245)
        return torch.add(critical_exp, 2.81)

    def get_dose_filters(self, tilt_index, lattice, Apix):
        D = lattice.D

        N = len(tilt_index)
        freqs = lattice.freqs2d / Apix  # D/A
        x = freqs[..., 0]
        y = freqs[..., 1]
        s2 = x ** 2 + y ** 2
        s = torch.sqrt(s2)

        cumulative_dose = self.tilt_numbers[tilt_index] * self.dose_per_tilt
        cd_tile = torch.repeat_interleave(cumulative_dose, D * D).view(N, -1)

        ce = self.critical_exposure(s).to(self.device)
        ce_tile = ce.repeat(N, 1)

        oe_tile = ce_tile * 2.51284  # Optimal exposure
        oe_mask = (cd_tile < oe_tile).long()

        freq_correction = torch.exp(-0.5 * cd_tile / ce_tile)
        freq_correction = torch.mul(freq_correction, oe_mask)
        angle_correction = torch.cos(self.tilt_angles[tilt_index] * np.pi / 180)
        ac_tile = torch.repeat_interleave(angle_correction, D * D).view(N, -1)

        return torch.mul(freq_correction, ac_tile).float()

    def optimal_exposure(self, freq):
        return 2.51284 * self.critical_exposure(freq)

    def get_tilting_func(self):
        def tilting_func(rots):
            """
            rots: [..., 3, 3]

            output: [..., n_tilts, 3, 3]
            """
            tilts = self.tilt_rots.to(rots.device)  # n_tilts, 3, 3
            return torch.sum(tilts[..., None] * rots[..., None, None, :, :], -2)

        return tilting_func


class DataShuffler:
    def __init__(
        self, dataset: ImageDataset, batch_size, buffer_size, dtype=np.float32
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.dtype = dtype
        assert self.buffer_size % self.batch_size == 0, (
            self.buffer_size,
            self.batch_size,
        )  # FIXME
        self.batch_capacity = self.buffer_size // self.batch_size
        assert self.buffer_size <= self.dataset.N, (
            self.buffer_size,
            self.dataset.N,
        )
        self.ntilts = getattr(dataset, "ntilts", 1)

    def __iter__(self):
        return _DataShufflerIterator(self)


class _DataShufflerIterator:
    def __init__(self, shuffler: DataShuffler):
        self.dataset = shuffler.dataset
        self.buffer_size = shuffler.buffer_size
        self.batch_size = shuffler.batch_size
        self.batch_capacity = shuffler.batch_capacity
        self.dtype = shuffler.dtype
        self.ntilts = shuffler.ntilts

        self.buffer = np.empty(
            (self.buffer_size, self.ntilts, self.dataset.D - 1, self.dataset.D - 1), dtype=self.dtype
        )
        self.tilt_index_buffer = np.full(
            (self.buffer_size, self.ntilts), -1, dtype=np.int64
        )
        self.index_buffer = np.full((self.buffer_size,), -1, dtype=np.int64)
        self.rot_buffer = np.empty(
            (self.buffer_size, self.ntilts, 3, 3), dtype=np.float32
        )
        if self.dataset.has_trans:
            self.trans_buffer = np.empty(
                (self.buffer_size, self.ntilts, 2), dtype=np.float32
            )
        self.num_batches = int(np.ceil(
            self.dataset.N / self.batch_size
        ))
        self.chunk_order = torch.randperm(self.num_batches)
        self.count = 0
        self.flush_remaining = -1  # at the end of the epoch, got to flush the buffer

        # pre-fill
        logger.info("Pre-filling data shuffler buffer...")
        for i in range(self.batch_capacity):
            chunk, maybe_tilt_indices, chunk_indices, chunk_rots, chunk_trans = self._get_next_chunk()
            self.buffer[i * self.batch_size : (i + 1) * self.batch_size] = chunk
            if maybe_tilt_indices is not None:
                self.tilt_index_buffer[
                i * self.batch_size: (i + 1) * self.batch_size
                ] = maybe_tilt_indices
            self.index_buffer[
                i * self.batch_size : (i + 1) * self.batch_size
            ] = chunk_indices
            self.rot_buffer[i * self.batch_size : (i + 1) * self.batch_size] = chunk_rots
            if self.dataset.has_trans:
                self.trans_buffer[i * self.batch_size : (i + 1) * self.batch_size] = chunk_trans

        logger.info(f"Filled buffer with {self.buffer_size} images "
                    f"({self.batch_capacity} contiguous chunks).")

    def _get_next_chunk(self):
        chunk_idx = int(self.chunk_order[self.count])
        self.count += 1
        start = chunk_idx * self.batch_size
        stop = (chunk_idx + 1) * self.batch_size
        if stop <= self.dataset.N:
            particles, maybe_tilt_indices = self.dataset.get_slice(
                start, stop
            )
            particles = particles.reshape(
                self.batch_size, self.ntilts, *particles.shape[1:]
            )
            particle_indices = np.arange(
                start, stop
            )
            tilt_indices = maybe_tilt_indices.reshape(
                    self.batch_size, self.ntilts
                ) if maybe_tilt_indices is not None else particle_indices
        else:
            stop_1 = self.dataset.N
            stop_2 = stop - self.dataset.N
            particles_1, maybe_tilt_indices_1 = self.dataset.get_slice(
                start, stop_1
            )
            particles_2, maybe_tilt_indices_2 = self.dataset.get_slice(
                0, stop_2
            )
            particles = np.concatenate((particles_1, particles_2), axis=0)
            particles = particles.reshape(
                self.batch_size, self.ntilts, *particles.shape[1:]
            )
            particle_indices = np.concatenate(
                (np.arange(start, stop_1), np.arange(0, stop_2)),
                axis=0
            )
            tilt_indices = np.concatenate((maybe_tilt_indices_1, maybe_tilt_indices_2), axis=0).reshape(
                self.batch_size, self.ntilts
            ) if maybe_tilt_indices_1 is not None else particle_indices
        rots = self.dataset.rot_gt[tilt_indices]
        trans = self.dataset.trans_gt[tilt_indices] if self.dataset.has_trans else None
        return particles, tilt_indices, particle_indices, rots, trans

    def __iter__(self):
        return self

    def __next__(self):
        """Returns a batch of images, and the indices of those images in the dataset.

        The buffer starts filled with `batch_capacity` random contiguous chunks.
        Each time a batch is requested, `batch_size` random images are selected from the buffer,
        and refilled with the next random contiguous chunk from disk.

        Once all the chunks have been fetched from disk, the buffer is randomly permuted and then
        flushed sequentially.
        """
        if self.count == self.num_batches and self.flush_remaining == -1:
            logger.info(
                "Finished fetching chunks. Flushing buffer for remaining batches..."
            )
            # since we're going to flush the buffer sequentially, we need to shuffle it first
            perm = np.random.permutation(self.buffer_size)
            self.buffer = self.buffer[perm]
            self.index_buffer = self.index_buffer[perm]
            self.flush_remaining = self.buffer_size

        if self.flush_remaining != -1:
            # we're in flush mode, just return chunks out of the buffer
            assert self.flush_remaining % self.batch_size == 0
            if self.flush_remaining == 0:
                raise StopIteration()
            particles = self.buffer[
                self.flush_remaining - self.batch_size : self.flush_remaining
            ]
            tilt_indices = self.tilt_index_buffer[
                           self.flush_remaining - self.batch_size: self.flush_remaining
                           ]
            particle_indices = self.index_buffer[
                self.flush_remaining - self.batch_size : self.flush_remaining
            ]
            rots = self.rot_buffer[
                   self.flush_remaining - self.batch_size: self.flush_remaining
            ]
            trans = self.trans_buffer[
                    self.flush_remaining - self.batch_size: self.flush_remaining
            ] if self.dataset.has_trans else None
            self.flush_remaining -= self.batch_size
        else:
            indices = np.random.choice(
                self.buffer_size, size=self.batch_size, replace=False
            )
            particles = self.buffer[indices]
            tilt_indices = self.tilt_index_buffer[indices]
            particle_indices = self.index_buffer[indices]
            rots = self.rot_buffer[indices]
            trans = self.trans_buffer[indices] if self.dataset.has_trans else None
            # start_loading = time.time()
            chunk, maybe_tilt_indices, chunk_indices, chunk_rots, chunk_trans = self._get_next_chunk()
            # log(f'Loading: {time.time() - start_loading}')
            self.buffer[indices] = chunk
            if maybe_tilt_indices is not None:
                self.tilt_index_buffer[indices] = maybe_tilt_indices
            self.index_buffer[indices] = chunk_indices
            self.rot_buffer[indices] = chunk_rots
            if self.dataset.has_trans:
                self.trans_buffer[indices] = chunk_trans

        particles = torch.from_numpy(particles).float()
        tilt_indices = torch.from_numpy(tilt_indices).long()
        particle_indices = torch.from_numpy(particle_indices).long()

        # start_processing = time.time()
        particles = particles.reshape(-1, *particles.shape[-2:])
        r_particles, particles = self.dataset._process(particles)
        # log(f'Processing: {time.time() - start_processing}')

        rots = torch.from_numpy(rots).float()

        if self.dataset.subtomogram_averaging:
            particles = particles.reshape(-1, self.ntilts, *particles.shape[-2:])
            tilt_indices = tilt_indices.reshape(-1, self.ntilts)
            r_particles = r_particles.reshape(-1, self.ntilts, *r_particles.shape[-2:])
            rots = rots.reshape(-1, self.ntilts, 3, 3)
        else:
            particles = particles.reshape(-1, *particles.shape[-2:])
            tilt_indices = tilt_indices.reshape(-1)
            r_particles = r_particles.reshape(-1, *r_particles.shape[-2:])
            rots = rots.reshape(-1, 3, 3)

        in_dict = {
            'y': particles,  # batch_size(, n_tilts), D, D
            'index': particle_indices,  # batch_size
            'tilt_index': tilt_indices.reshape(-1),  # batch_size * n_tilts
            'y_real': r_particles,  # batch_size(, n_tilts), D - 1, D - 1
            'R': rots  # batch_size(, n_tilts), 3, 3
        }

        if self.dataset.has_trans:
            if self.dataset.subtomogram_averaging:
                in_dict['t'] = torch.from_numpy(trans).float().reshape(-1, self.ntilts, 2)  # batch_size, n_tilts, 2
            else:
                in_dict['t'] = torch.from_numpy(trans).float().reshape(-1, 2)  # batch_size, 2

        return in_dict


def make_dataloader(
    data: ImageDataset, *, batch_size: int, num_workers: int = 0, shuffler_size: int = 0
):
    if shuffler_size > 0:
        assert data.lazy, "Only enable a data shuffler for lazy loading"
        return DataShuffler(data, batch_size=batch_size, buffer_size=shuffler_size)
    else:
        return DataLoader(
            data,
            num_workers=num_workers,
            sampler=BatchSampler(
                RandomSampler(data), batch_size=batch_size, drop_last=False
            ),
            batch_size=None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )