"""
Lattice object
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Lattice:
    def __init__(self, resolution, extent=0.5, ignore_dc=True, device=None):
        """
        coords: [resolution**2, 3]
        """
        assert resolution % 2 == 1, "Lattice size must be odd"

        x0, x1 = np.meshgrid(
            np.linspace(-extent, extent, resolution, endpoint=True),
            np.linspace(-extent, extent, resolution, endpoint=True)
            )
        coords = np.stack([x0.ravel(), x1.ravel(), np.zeros(resolution ** 2)],
                          1).astype(np.float32)

        self.coords = torch.tensor(coords, device=device)
        self.extent = extent
        self.D = resolution
        self.D2 = int(resolution / 2)

        self.center = torch.tensor([0., 0.], device=device)

        self.square_mask = {}
        self.circle_mask = {}

        self.freqs2d = self.coords[:, 0:2] / extent / 2

        self.ignore_DC = ignore_dc
        self.device = device

    def get_downsample_coords(self, d):
        assert d % 2 == 1, f"downsample dimension {d=} must be odd!"

        extent = self.extent * (d - 1) / (self.D - 1)
        x0, x1 = np.meshgrid(np.linspace(-extent, extent, d, endpoint=True),
                             np.linspace(-extent, extent, d, endpoint=True))
        coords = np.stack(
            [x0.ravel(), x1.ravel(), np.zeros(d ** 2)], 1).astype(np.float32)

        return torch.tensor(coords, device=self.device)

    def get_square_lattice(self, side_length):
        b, e = self.D2 - side_length, self.D2 + side_length + 1
        center_lattice = self.coords.view(
            self.D, self.D, 3)[b:e, b:e, :].contiguous().view(-1, 3)

        return center_lattice

    def get_square_mask(self, side_length):
        """Return a centered square lattice binary mask for self.coords."""
        if side_length in self.square_mask:
            return self.square_mask[side_length]

        mask_len = 2 * side_length + 1
        if mask_len <= self.D:
            raise ValueError(f"Mask with size {side_length} too large for "
                             f"lattice with size {self.D=}")

        logger.info(f"Using square lattice of size {mask_len}x{mask_len}")
        b, e = self.D2 - side_length, self.D2 + side_length
        c1 = self.coords.view(self.D, self.D, 3)[b, b]
        c2 = self.coords.view(self.D, self.D, 3)[e, e]
        m1 = self.coords[:, 0] >= c1[0]
        m2 = self.coords[:, 0] <= c2[0]
        m3 = self.coords[:, 1] >= c1[1]
        m4 = self.coords[:, 1] <= c2[1]
        mask = m1 * m2 * m3 * m4
        self.square_mask[side_length] = mask

        if self.ignore_DC:
            raise NotImplementedError

        return mask

    def rotate(self, images, thetas):
        """
        images: [batch_size, h, w]
        thetas: [n_thetas] in radians

        output: [batch_size, n_thetas, h, w]
        """
        n_thetas = len(thetas)
        images = images.expand(n_thetas, *images.shape)
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        rot = torch.stack([cos, sin, -sin, cos], 1).view(-1, 2, 2)
        grid = self.coords[:, 0:2] / self.extent @ rot  # grid between -1 and 1
        grid = grid.view(n_thetas, self.D, self.D, 2)
        offset = self.center - grid[:, self.D2, self.D2]
        grid += offset[:, None, None, :]
        rotated = F.grid_sample(images, grid, align_corners=False)

        return rotated.transpose(0, 1)

    def translate_ft(self, img, t, mask=None):
        """
        Translate an image by phase shifting its Fourier transform
        F'(k) = exp(-2*pi*k*x0)*F(k)

        img: FT of image [batch_size, img_dims, 2]
        t: shift in pixels [batch_size, n_trans, 2]
        mask: Mask for lattice coords [img_dims, 1]

        output: Shifted images [batch_size, n_trans, img_dims x 2]

        img_dims can either be 2D or 1D (unraveled image)
        """
        coords = self.freqs2d if mask is None else self.freqs2d[mask]
        img = img[:, None]
        t = t[..., None]
        phase = coords @ t * -2.0 * np.pi
        phase = phase[..., 0]
        c = torch.cos(phase)
        s = torch.sin(phase)

        translated_ft = torch.stack(
            [img[..., 0] * c - img[..., 1] * s, img[..., 0] * s + img[..., 1]
             * c], -1
            )

        return translated_ft

    def translate_ht(self, img, t, freqs2d=None, mask=None):
        """
        Translate an image by phase shifting its Hartley transform
        H'(k) = cos(2*pi*k*t0)H(k) + sin(2*pi*k*t0)H(-k)

        img: HT of image [B, img_dims]
        t: shift in pixels [batch_size, n_trans, 2]
        mask: Mask for lattice coords [img_dims, 1]

        output: Shifted images [batch_size, n_trans, img_dims]

        img must be 1D unraveled image, symmetric around DC component
        """
        freqs = freqs2d if freqs2d is not None else self.freqs2d
        coords = freqs if mask is None else freqs[mask]
        img = img[:, None]
        t = t[..., None]
        phase = coords @ t * 2 * np.pi
        phase = phase[..., 0]
        c = torch.cos(phase)
        s = torch.sin(phase)

        return c * img + s * img[:, :, torch.arange(len(coords) - 1, -1, -1)]
