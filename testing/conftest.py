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


class OutDir:
    """Utility class for representing the output of a cryoDRGN experiment."""

    def __init__(self, outpath: Path):
        self.basepath = outpath
        self.outpath = outpath / 'out'

    def hvolm_hash(self, epoch: int) -> float:
        weights_fl = self.outpath / f"weights.{epoch}.pkl"
        hvolm = torch.load(weights_fl)['hypervolume_state_dict']

        return round(hvolm['mlp.main.0.weight'].sum().item(), 3)

    def conf_hash(self, epoch: int) -> float:
        conf = utils.load_pkl(self.outpath / f"conf.{epoch}.pkl")
        zdim = conf.shape[1]

        rep_vec = np.tile([1, -1], zdim // 2 + zdim % 2)[:zdim]
        return round((conf @ rep_vec).sum(), 3)

    @property
    def configs(self) -> dict[str, Any]:
        with open(os.path.join(self.basepath, "configs.yaml"), 'r') as f:
            configs = yaml.safe_load(f)

        return configs


@pytest.fixture
def outdir(request, tmp_path) -> OutDir:
    """Set up and tear down an output folder with a set of configurations."""

    orig_paths = os.environ.get('DRGNAI_DATASETS')
    os.environ['DRGNAI_DATASETS'] = str(DATASET_DIR)

    if hasattr(request, 'param') and request.param:
        with open(tmp_path / 'configs.yaml', 'w') as f:
            yaml.dump(request.param, f, sort_keys=False)

    yield OutDir(tmp_path)

    if orig_paths is not None:
        os.environ['DRGNAI_DATASETS'] = orig_paths
