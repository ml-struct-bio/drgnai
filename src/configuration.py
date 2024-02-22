"""Parsing and storing configuration parameters for models."""

import os
import sys
import yaml
import numpy as np
from collections import OrderedDict
from abc import ABC

CONFIG_DIR = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'configs')


class _BaseConfigurations(ABC):
    """Base class for sets of model configuration parameters."""

    # a parameter belongs to this set if and only if it has a default value
    # defined in this dictionary, ordering makes e.g. printing easier for user
    defaults = OrderedDict()

    @classmethod
    @property
    def parameters(cls) -> list:
        return list(cls.defaults.keys())

    def __init__(self, config_vals: dict) -> None:
        for key in config_vals:
            if key not in self.defaults:
                raise ValueError("Unrecognized configuration "
                                 f"parameter `{key}`!")

        # an attribute is created for every entry in the defaults dictionary
        for key, value in self.defaults.items():
            if key in config_vals:
                setattr(self, key, config_vals[key])

            # if not in given parameters, use defaults
            else:
                setattr(self, key, value)

    def __iter__(self):
        return iter((par, getattr(self, par)) for par in self.parameters)

    def __str__(self):
        return '\n'.join([f"{par}{str(val):>20}" for par, val in self])

    def write(self, fl: str) -> None:
        """Saving configurations to file using the original order."""

        with open(fl, 'w') as f:
            yaml.dump(dict(self), f, default_flow_style=False, sort_keys=False)


class TrainingConfigurations(_BaseConfigurations):

    defaults = OrderedDict({
        'outdir': None, 'quick_config': None,

        # dataset
        'particles': None, 'ctf': None, 'pose': None, 'dataset': None,
        'server': None, 'datadir': None, 'ind': None,
        'labels': None, 'relion31': False, 'no_trans': False,

        # initialization
        'use_gt_poses': False, 'refine_gt_poses': False,
        'use_gt_trans': False, 'load': None, 'initial_conf': None,

        # logging
        'log_interval': 10000, 'log_heavy_interval': 5, 'verbose_time': False,

        # data loading
        'shuffle': True, 'lazy': False, 'num_workers': 2, 'max_threads': 16,
        'fast_dataloading': False, 'shuffler_size': 32768,
        'batch_size_known_poses': 32, 'batch_size_hps': 8,
        'batch_size_sgd': 256,

        # optimizers
        'hypervolume_optimizer_type': "adam",
        'pose_table_optimizer_type': "adam",
        'conf_table_optimizer_type': "adam",
        'conf_encoder_optimizer_type': "adam",
        'lr': 1.0e-4, 'lr_pose_table': 1.0e-3,
        'lr_conf_table': 1.0e-2, 'lr_conf_encoder': 1.0e-4, 'wd': 0.0,

        # scheduling
        'n_imgs_pose_search': 500000, 'epochs_sgd': 100, 'pose_only_phase': 0,

        # masking
        'output_mask': "circ", 'add_one_frequency_every': 100000,
        'n_frequencies_per_epoch': 10, 'max_freq': None,
        'window_radius_gt_real': 0.85, 'l_start_fm': 12,

        # loss
        'beta_conf': 0.0, 'trans_l1_regularizer': 0.0,
        'l2_smoothness_regularizer': 0.0,

        # conformations
        'variational_het': False, 'z_dim': 4, 'std_z_init': 0.1,
        'use_conf_encoder': False, 'depth_cnn': 5, 'channels_cnn': 32,
        'kernel_size_cnn': 3, 'resolution_encoder': None,

        # hypervolume
        'explicit_volume': False, 'hypervolume_layers': 3,
        'hypervolume_dim': 256, 'pe_type': "gaussian", 'pe_dim': 64,
        'feat_sigma': 0.5, 'hypervolume_domain': "hartley",
        'pe_type_conf': None,

        # pre-training
        'n_imgs_pretrain': 10000, 'pretrain_with_gt_poses': False,

        # pose search
        'l_start': 12, 'l_end': 32, 'n_iter': 4, 't_extent': 20.0,
        't_n_grid': 7, 't_x_shift': 0.0, 't_y_shift': 0.0,
        'no_trans_search_at_pose_search': False, 'n_kept_poses': 8,
        'base_healpy': 2,

        # subtomogram averaging
        'subtomogram_averaging': False,
        'n_tilts': 11,
        'dose_per_tilt': 2.93,
        'angle_per_tilt': 3.0,
        'n_tilts_pose_search': 11,
        'average_over_tilts': False,
        'tilt_axis_angle': 0.0,
        'dose_exposure_correction': True,

        # others
        'seed': -1,
        'color_palette': None,
        'test_installation': False,
        })

    quick_defaults = {
        'capture_setup': {
            'spa': dict(),
            'et': {'subtomogram_averaging': True, 'fast_dataloading': True,
                   'shuffler_size': 0, 'num_workers': 0, 't_extent': 0.0,
                   'batch_size_known_poses': 8, 'batch_size_sgd': 32,
                   'n_imgs_pose_search': 150000, 'pose_only_phase': 50000,
                   'lr_pose_table': 1.0e-5}
            },

        'reconstruction_type': {
            'homo': {'z_dim': 0}, 'het': dict()},

        'pose_estimation': {
            'abinit': dict(),
            'refine': {'refine_gt_poses': True, 'lr_pose_table': 1.0e-4},
            'fixed': {'use_gt_poses': True}
            },

        'conf_estimation': {
            None: dict(),
            'autodecoder': dict(),
            'refine': dict(),
            'encoder': {'use_conf_encoder': True}
            }
        }

    def __init__(self, config_vals: dict):
        super().__init__(config_vals)

        if self.test_installation:
            print('Installation was successful!')
            sys.exit()

        if self.seed < 0:
            self.seed = np.random.randint(0, 10000)

        # process the quick_config parameter
        if self.quick_config is not None:
            for key, value in self.quick_config.items():
                if key not in self.quick_defaults:
                    raise ValueError("Unrecognized parameter "
                                     f"shortcut field `{key}`!")

                if value not in self.quick_defaults[key]:
                    raise ValueError("Unrecognized parameter shortcut label "
                                     f"`{value}` for field `{key}`!")

                for _key, _value in self.quick_defaults[key][value].items():
                    if _key not in self.defaults:
                        raise ValueError("Unrecognized configuration "
                                         f"parameter `{key}`!")

                    # given parameters have priority
                    if _key not in config_vals:
                        setattr(self, _key, _value)

        if not self.outdir:
            raise ValueError("Must specify an outdir!")

        if self.explicit_volume and self.z_dim >= 1:
            raise ValueError("Explicit volumes do not support "
                             "heterogeneous reconstruction.")

        if self.dataset is None:
            if self.particles is None:
                raise ValueError("As dataset was not specified, please "
                                 "specify particles!")

            if self.ctf is None:
                raise ValueError("As dataset was not specified, please "
                                 "specify ctf!")

        if self.hypervolume_optimizer_type not in {'adam'}:
            raise ValueError("Invalid value "
                             f"`{self.hypervolume_optimizer_type}` "
                             "for hypervolume_optimizer_type!")

        if self.pose_table_optimizer_type not in {'adam', 'lbfgs'}:
            raise ValueError("Invalid value "
                             f"`{self.pose_table_optimizer_type}` "
                             "for pose_table_optimizer_type!")

        if self.conf_table_optimizer_type not in {'adam', 'lbfgs'}:
            raise ValueError("Invalid value "
                             f"`{self.conf_table_optimizer_type}` "
                             "for conf_table_optimizer_type!")

        if self.conf_encoder_optimizer_type not in {'adam'}:
            raise ValueError("Invalid value "
                             f"`{self.conf_encoder_optimizer_type}` "
                             "for conf_encoder_optimizer_type!")

        if self.output_mask not in {'circ', 'frequency_marching'}:
            raise ValueError("Invalid value "
                             f"{self.output_mask} for output_mask!")

        if self.pe_type not in {'gaussian'}:
            raise ValueError(f"Invalid value {self.pe_type} for pe_type!")

        if self.pe_type_conf not in {None, 'geom'}:
            raise ValueError(f"Invalid value {self.pe_type_conf} "
                             "for pe_type_conf!")

        if self.hypervolume_domain not in {'hartley'}:
            raise ValueError(f"Invalid value {self.hypervolume_domain} "
                             "for hypervolume_domain.")

        if self.n_imgs_pose_search < 0:
            raise ValueError("n_imgs_pose_search must be greater than 0!")

        if self.use_conf_encoder and self.initial_conf:
            raise ValueError("Conformations cannot be initialized "
                             "when using an encoder!")

        if self.use_gt_trans and self.pose is None:
            raise ValueError("Poses must be specified to use GT translations!")

        if self.refine_gt_poses:
            self.n_imgs_pose_search = 0

            if self.pose is None:
                raise ValueError("Initial poses must be specified "
                                 "to be refined!")

        if self.subtomogram_averaging:
            self.fast_dataloading = True

            # TODO: Implement conformation encoder for subtomogram averaging.
            if self.use_conf_encoder:
                raise ValueError("Conformation encoder is not implemented "
                                 "for subtomogram averaging!")

            # TODO: Implement translation search for subtomogram averaging.
            if not (self.use_gt_poses or self.use_gt_trans
                    or self.t_extent == 0.):
                raise ValueError("Translation search is not implemented "
                                 "for subtomogram averaging!")

            if self.average_over_tilts and self.n_tilts_pose_search % 2 == 0:
                raise ValueError("n_tilts_pose_search must be odd "
                                 "to use average_over_tilts!")

            if self.n_tilts_pose_search > self.n_tilts:
                raise ValueError("n_tilts_pose_search must be "
                                 "smaller than n_tilts!")
        if self.use_gt_poses:
            # "poses" include translations
            self.use_gt_trans = True

            if self.pose is None:
                raise ValueError("Ground truth poses must be specified!")

        if self.no_trans:
            self.t_extent = 0.
        if self.t_extent == 0.:
            self.t_n_grid = 1

        if self.dataset:
            with open(os.environ.get("DRGNAI_DATASETS"), 'r') as f:
                paths = yaml.safe_load(f)

            self.particles = paths[self.dataset]['particles']
            self.ctf = paths[self.dataset]['ctf']

            if self.pose is None and 'pose' in paths[self.dataset]:
                self.pose = paths[self.dataset]['pose']
            if 'datadir' in paths[self.dataset]:
                self.datadir = paths[self.dataset]['datadir']
            if 'labels' in paths[self.dataset]:
                self.labels = paths[self.dataset]['labels']
            if self.ind is None and 'ind' in paths[self.dataset]:
                self.ind = paths[self.dataset]['ind']
            if 'dose_per_tilt' in paths[self.dataset]:
                self.dose_per_tilt = paths[self.dataset]['dose_per_tilt']


class AnalysisConfigurations(_BaseConfigurations):

    defaults = {'epoch': -1, 'skip_umap': False, 'pc': 2, 'n_per_pc': 10,
                'ksample': 20, 'seed': -1, 'invert': True,
                'sample_z_idx': None, 'trajectory_1d': None,
                'direct_traversal_txt': None, 'z_values_txt': None}
