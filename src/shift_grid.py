import numpy as np


def grid_1d(resol, extent, ngrid, shift=0):
    Npix = ngrid * 2 ** resol
    dt = 2 * extent / Npix
    grid = np.arange(Npix, dtype=np.float32) * dt + dt / 2 - extent + shift
    return grid


def grid_2d(resolution, extent, n_grid, x_shift=0, y_shift=0):
    x = grid_1d(resolution, extent, n_grid, shift=x_shift)
    y = grid_1d(resolution, extent, n_grid, shift=y_shift)
    # convention: x is fast dim, y is slow dim
    grid = np.stack(np.meshgrid(x, y), -1)
    return grid.reshape(-1, 2)


def base_shift_grid(resolution, extent, n_grid, x_shift=0, y_shift=0):
    return grid_2d(resolution, extent, n_grid, x_shift, y_shift)


def get_1d_neighbor(mini, cur_res, extent, n_grid):
    n_pix = n_grid * 2 ** (cur_res + 1)
    dt = 2 * extent / n_pix
    ind = np.array([2 * mini, 2 * mini + 1], dtype=np.float32)
    return dt * ind + dt / 2 - extent, ind


def get_base_ind(ind, n_grid):
    xi = ind % n_grid
    yi = ind // n_grid
    return np.stack((xi, yi), axis=1)


def get_neighbor(xi, yi, cur_res, extent, n_grid):
    """
    Return the 4 nearest neighbors at the next resolution level
    """
    x_next, xii = get_1d_neighbor(xi, cur_res, extent, n_grid)
    y_next, yii = get_1d_neighbor(yi, cur_res, extent, n_grid)
    t_next = np.stack(np.meshgrid(x_next, y_next), -1).reshape(-1, 2)
    ind_next = np.stack(np.meshgrid(xii, yii), -1).reshape(-1, 2)
    return t_next, ind_next
