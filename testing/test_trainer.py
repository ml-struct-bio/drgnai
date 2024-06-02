"""Unit tests of model trainer behaviour."""

import pytest
import os
from cryodrgnai.reconstruct import ModelTrainer

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
DATASET_DIR = os.path.join(DATA_DIR, "data-paths.yaml")
TEST_ARGS = {
    'num_workers': 1, 'hypervolume_dim': 4, 'pe_dim': 4, 'hypervolume_layers': 2
}


@pytest.mark.parametrize(
    "configs_outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "toy_projections.mrcs"),
                'ctf': os.path.join(DATA_DIR, "test_ctf.pkl"),
                'pose': os.path.join(DATA_DIR, "toy_rot_trans.pkl"), 'ind': 10,
                'n_imgs_pose_search': 20, 'n_imgs_pretrain': 20, 'epochs_sgd': 2,
                'batch_size_known_poses': 5, 'shuffle': False,
                't_extent': 4.0, 't_n_grid': 2, 'seed': 307, 'lazy': lazy_load,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'fixed',
                                 'reconstruction_type': 'homo'}})
        for lazy_load in (False, True)
        ],
    ids=("toy-notlazy", "toy-lazy"),
    indirect=True
    )
class TestShuffling:

    def test_no_shuffling(self, configs_outdir):
        """Run training and analyses."""
        cfgs = configs_outdir.load_configs()
        trainer = ModelTrainer(str(configs_outdir.basepath), cfgs)
        trainer.train()
        assert trainer.in_dict_last['index'].tolist() == [5, 6, 7, 8, 9]

    def test_fixed_shuffling(self, configs_outdir):
        cfgs = configs_outdir.load_configs()
        cfgs['shuffle'] = True

        trainer = ModelTrainer(str(configs_outdir.basepath), cfgs)
        trainer.train()
        assert trainer.in_dict_last['index'].tolist() == [9, 6, 7, 3, 8]

        trainer = ModelTrainer(str(configs_outdir.basepath), cfgs)
        trainer.train()
        assert trainer.in_dict_last['index'].tolist() == [9, 6, 7, 3, 8]

    def test_change_batch_size(self, configs_outdir):
        cfgs = configs_outdir.load_configs()
        cfgs['batch_size_known_poses'] = 4
        cfgs['shuffle'] = True

        trainer = ModelTrainer(str(configs_outdir.basepath), cfgs)
        trainer.train()
        assert trainer.in_dict_last['index'].tolist() == [3, 8]


@pytest.mark.parametrize(
    "outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "toy_projections.mrcs"),
                'ctf': os.path.join(DATA_DIR, "test_ctf.pkl"),
                'pose': os.path.join(DATA_DIR, "toy_rot_trans.pkl"), 'ind': 3,
                'n_imgs_pose_search': 5, 'n_imgs_pretrain': 6, 'epochs_sgd': 1,
                't_extent': 4.0, 't_n_grid': 2})
        ],
    ids=("toy", ),
    indirect=True
    )
class TestOldOutputs:

    def test_auto_folder_renaming(self, outdir):
        """Run training and analyses."""
        cfgs = outdir.params
        cfgs['quick_config'] = {'capture_setup': 'spa',
                                'conf_estimation': 'autodecoder',
                                'pose_estimation': 'fixed',
                                'reconstruction_type': 'het'}
        cfgs['z_dim'] = 2

        ModelTrainer(str(outdir.basepath), cfgs).train()
        out_files = {'out'}
        assert set(os.listdir(outdir.basepath)) == out_files

        cfgs['quick_config'] = {'capture_setup': 'spa',
                                'conf_estimation': 'autodecoder',
                                'pose_estimation': 'refine',
                                'reconstruction_type': 'homo'}
        cfgs['z_dim'] = 0

        ModelTrainer(str(outdir.basepath), cfgs).train()
        out_files |= {"old-out_000_fixed-het2"}
        assert set(os.listdir(outdir.basepath)) == out_files

        cfgs['quick_config'] = {'capture_setup': 'spa',
                                'conf_estimation': 'autodecoder',
                                'pose_estimation': 'abinit',
                                'reconstruction_type': 'homo'}

        ModelTrainer(str(outdir.basepath), cfgs).train()
        out_files |= {"old-out_001_refine-homo"}
        assert set(os.listdir(outdir.basepath)) == out_files

        for i in range(3):
            ModelTrainer(str(outdir.basepath), cfgs).train()
            out_files |= {f"old-out_00{i + 2}_abinit-homo"}
            assert set(os.listdir(outdir.basepath)) == out_files
