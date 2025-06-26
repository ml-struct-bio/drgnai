"""Fixtures and other resources used across many tests."""

import os
from pathlib import Path
import pytest
from typing import Any
import yaml
import numpy as np
import torch
from cryodrgn import utils

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
DATASET_DIR = os.path.join(DATA_DIR, "data-paths.yaml")


def hash_mat(mat) -> float:
    """Get quick hash value for a two-dimensional array."""
    mat_ydim = mat.shape[1]
    rep_vec = np.tile([1, -1], mat_ydim // 2 + mat_ydim % 2)[:mat_ydim]
    if isinstance(mat, torch.Tensor):
        mat = mat.cpu()

    return (mat @ rep_vec).sum()


class OutDir:
    """Utility class for representing the output of a cryoDRGN experiment."""

    def __init__(self, outpath: Path, params: dict[str, Any]) -> None:
        self.basepath = outpath
        self.outpath = outpath / 'out'
        self.params = params

    def weights_hash(self, epoch: int) -> float:
        """Get a hash of the saved model weights for a given epoch."""
        checkpoint = torch.load(self.outpath / f"weights.{epoch}.pkl")

        hash_val = sum(hash_mat(model_vals)
                       for model_k, model_vals in checkpoint['model_state_dict'].items()
                       if model_k[-6:] == 'weight')
        hash_val += sum(hash_mat(model_vals)
                        for model_k, model_vals
                        in checkpoint['hypervolume_state_dict'].items()
                        if model_k[-6:] == 'weight')

        return round(hash_val.item(), 3)

    def pose_hash(self, epoch: int) -> float:
        """Get a hash of the saved poses for a given epoch."""
        rot, trans = utils.load_pkl(self.outpath / f"pose.{epoch}.pkl")
        rot_hash, trans_hash = hash_mat(rot), hash_mat(trans)

        return round((rot_hash + trans_hash).item(), 3)

    def conf_hash(self, epoch: int) -> float:
        """Get a hash of the saved latent conformations for a given epoch."""
        return round(
            hash_mat(utils.load_pkl(self.outpath / f"conf.{epoch}.pkl")).item(), 3)

    def output_hash(self, epoch: int) -> float:
        """Get a hash of the saved training model outputs for a given epoch."""
        hash_val = 0.
        out_files = os.listdir(self.outpath)

        if f"weights.{epoch}.pkl" in out_files:
            hash_val += self.weights_hash(epoch)
        if f"pose.{epoch}.pkl" in out_files:
            hash_val += self.pose_hash(epoch)
        if f"conf.{epoch}.pkl" in out_files:
            hash_val += self.conf_hash(epoch)

        return round(hash_val, 3)

    def load_configs(self) -> dict[str, Any]:
        """Load the input configurations for this experiment directory."""
        with open(os.path.join(self.basepath, "configs.yaml"), 'r') as f:
            configs = yaml.safe_load(f)

        return configs

    def save_configs(self, cfgs: dict[str, Any]) -> None:
        """Save or overwrite the input configurations for this experiment directory."""
        with open(os.path.join(self.basepath, "configs.yaml"), 'w') as f:
            yaml.dump(cfgs, f)


@pytest.fixture
def outdir(request, tmp_path) -> OutDir:
    """Set up and tear down an empty output folder."""

    orig_paths = os.environ.get('DRGNAI_DATASETS')
    os.environ['DRGNAI_DATASETS'] = str(DATASET_DIR)

    params = request.param if hasattr(request, 'param') and request.param else None
    yield OutDir(tmp_path, params)

    if orig_paths is not None:
        os.environ['DRGNAI_DATASETS'] = orig_paths


@pytest.fixture
def configs_outdir(request, tmp_path) -> OutDir:
    """Set up and tear down an output folder with a set of configurations."""

    orig_paths = os.environ.get('DRGNAI_DATASETS')
    os.environ['DRGNAI_DATASETS'] = str(DATASET_DIR)

    if hasattr(request, 'param') and request.param:
        params = request.param
        with open(tmp_path / 'configs.yaml', 'w') as f:
            yaml.dump(request.param, f, sort_keys=False)
    else:
        params = None

    yield OutDir(tmp_path, params)

    if orig_paths is not None:
        os.environ['DRGNAI_DATASETS'] = orig_paths
