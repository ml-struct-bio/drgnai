"""Unit and fidelity tests of the pipeline command line interface.

The hash values in these tests have been validated against Alex Levy's cryodrgnai
(commit d366c92 at https://github.com/ml-struct-bio/drgnai-internal); these tests thus
check if the outputs of the current version of drgnai (specifically, reconstructed
volumes and poses as well as model weights) match the outputs of this "reference"
version of cryodrgnai.

"""
import pytest
import os
from cryodrgnai.utils import run_command

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
DATASET_DIR = os.path.join(DATA_DIR, "data-paths.yaml")

TEST_ARGS = {
    'num_workers': 1, 'max_threads': 1, 'hypervolume_dim': 4, 'pe_dim': 4,
    'hypervolume_layers': 2, 'shuffle': False, 'lazy': False,
}


@pytest.mark.parametrize(
    "configs_outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "toy_projections.mrcs"),
                'ctf': os.path.join(DATA_DIR, "test_ctf.pkl"), 'ind': 5,
                'n_imgs_pose_search': 10,  'epochs_sgd': 2,
                't_extent': 4.0, 't_n_grid': 2,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'het'}})
        ],
    ids=("toy", ),
    indirect=True
    )
def test_no_seed(configs_outdir):
    """Create a toy dataset, run the pipeline, then restart from saved checkpoint."""
    out, err = run_command(f"drgnai train {configs_outdir.basepath}")
    assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
    assert 'Finished in ' in out, err

    assert (configs_outdir.outpath / 'weights.4.pkl').exists()
    assert (configs_outdir.outpath / 'analysis_4').exists()
    assert (configs_outdir.outpath / 'analysis_4' / 'z_pca_hexbin.png').exists()


@pytest.mark.parametrize(
    "configs_outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "toy_projections.mrcs"),
                'ctf': os.path.join(DATA_DIR, "test_ctf.pkl"), 'ind': 15,
                'n_imgs_pose_search': 30,  'epochs_sgd': 2,
                't_extent': 4.0, 't_n_grid': 2, "invert_data": False,
                'z_dim': 4, 'seed': 1701, 'log_heavy_interval': 2,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'het'}})
        ],
    ids=("toy", ),
    indirect=True
    )
def test_small_dataset_and_load(configs_outdir):
    """Create a toy dataset, run the pipeline, then restart from saved checkpoint."""
    out, err = run_command(f"drgnai train {configs_outdir.basepath} --no-analysis")
    assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
    assert 'Finished in ' in out, err

    # confirm that log_heavy_interval is working correctly
    assert (configs_outdir.outpath / 'weights.0.pkl').exists()
    assert (configs_outdir.outpath / 'weights.1.pkl').exists()
    assert (configs_outdir.outpath / 'weights.2.pkl').exists()
    assert not (configs_outdir.outpath / 'analysis_2').exists()
    assert not (configs_outdir.outpath / 'weights.3.pkl').exists()
    assert (configs_outdir.outpath / 'weights.4.pkl').exists(), err
    assert not (configs_outdir.outpath / 'analysis_4').exists(), err

    assert configs_outdir.output_hash(2) == 14.415
    assert configs_outdir.output_hash(4) == 13.535

    # manually create configurations file for run where we restart from checkpoint
    new_configs = configs_outdir.load_configs()
    new_configs['seed'] = 901
    old_lbl = 'old-out_000_abinit-het4'
    new_configs['load'] = os.path.abspath(
        os.path.join(configs_outdir.basepath, old_lbl, 'weights.4.pkl'))
    configs_outdir.save_configs(new_configs)

    out, err = run_command(f"drgnai train {configs_outdir.basepath}")
    assert set(os.listdir(configs_outdir.basepath)) == {
        old_lbl, 'out', 'configs.yaml'}, err
    assert 'Finished in ' in out

    assert (configs_outdir.outpath / f'weights.6.pkl').exists(), err
    assert (configs_outdir.outpath / 'analysis_6').exists(), err
    assert configs_outdir.output_hash(6) == 12.803


@pytest.mark.parametrize(
    "outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "hand.5.mrcs"),
                'ctf': os.path.join(DATA_DIR, "hand.5_ctf.pkl"), 'ind': ind,
                'n_imgs_pose_search': 10, 'epochs_sgd': 3, 'seed': 555,
                't_extent': 4.0, 't_n_grid': 2, 'log_heavy_interval': 5,
                'invert_data': False,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'homo'}})
        for ind in [None, 3, os.path.join(DATA_DIR, "hand-ind3.pkl")]
        ],
    ids=("hand_no-ind", "hand_int-ind", "hand_file-ind"),
    indirect=True
)
class TestSetupIntegration:
    """Running the setup command separately before training."""

    def test_configuration_keys(self, outdir):
        """Running setup with extra configuration keys."""
        args_str = " ".join([f"\'{k}={v}\'" for k, v in outdir.params.items()
                             if k != "quick_config"])
        recon_type = outdir.params['quick_config']['reconstruction_type']

        out, err = run_command(
            f"drgnai setup {outdir.basepath} --cfgs {args_str} "
            f"--capture={outdir.params['quick_config']['capture_setup']} "
            f"--pose-estim={outdir.params['quick_config']['pose_estimation']} "
            f"--conf-estim={outdir.params['quick_config']['conf_estimation']} "
            f"--reconstruction-type={recon_type} "
        )
        assert err == "", err

        out, err = run_command(f"drgnai train {outdir.basepath} --no-analysis")
        assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}, err
        assert 'Finished in ' in out, err

        assert (outdir.outpath / 'weights.5.pkl').exists()
        assert not (outdir.outpath / 'analysis_5').exists()
        assert outdir.output_hash(5) == {
            None: 10.572, 3: 4.68,
            os.path.join(DATA_DIR, "hand-ind3.pkl"): 17.189,
            }[outdir.params['ind']]


@pytest.mark.parametrize(
    "configs_outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "hand.5.mrcs"),
                'ctf': os.path.join(DATA_DIR, "hand.5_ctf.pkl"),
                'n_imgs_pose_search': 10, 'seed': 555,
                't_extent': 4.0, 't_n_grid': 2, 'log_heavy_interval': 2,
                'norm_mean': 0.0, 'norm_std': 2.0,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'homo'}})
        ],
    ids=("hand", ),
    indirect=True
    )
class TestHomogeneousTrain:
    """Running homogeneous reconstruction using command line arguments."""

    def test_reconstruction(self, configs_outdir):
        """Run just reconstruction without post-analysis."""
        cfgs = configs_outdir.load_configs()
        cfgs['epochs_sgd'] = 3
        configs_outdir.save_configs(cfgs)

        out, err = run_command(f"drgnai train {configs_outdir.basepath} --no-analysis")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
        assert 'Finished in ' in out, err

        assert (configs_outdir.outpath / 'weights.0.pkl').exists()
        assert (configs_outdir.outpath / 'weights.1.pkl').exists()
        assert (configs_outdir.outpath / 'weights.2.pkl').exists()
        assert not (configs_outdir.outpath / 'weights.3.pkl').exists()
        assert (configs_outdir.outpath / 'weights.4.pkl').exists()
        assert (configs_outdir.outpath / 'weights.5.pkl').exists()
        assert not (configs_outdir.outpath / 'analysis_5').exists()
        assert configs_outdir.output_hash(5) == 5.689

    def test_auto_load(self, configs_outdir):
        """Run some of the same epochs first; then use auto-restart to do the rest."""
        cfgs = configs_outdir.load_configs()
        cfgs['epochs_sgd'] = 1
        configs_outdir.save_configs(cfgs)

        out, err = run_command(f"drgnai train {configs_outdir.basepath}")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
        assert 'Finished in ' in out, err

        assert (configs_outdir.outpath / 'weights.0.pkl').exists()
        assert (configs_outdir.outpath / 'weights.1.pkl').exists()
        assert (configs_outdir.outpath / 'weights.2.pkl').exists()
        assert (configs_outdir.outpath / 'weights.3.pkl').exists()
        assert (configs_outdir.outpath / 'analysis_3').exists()
        assert not (configs_outdir.outpath / 'weights.4.pkl').exists()
        assert not (configs_outdir.outpath / 'weights.5.pkl').exists()
        assert not (configs_outdir.outpath / 'weights.6.pkl').exists()

        cfgs = configs_outdir.load_configs()
        cfgs['epochs_sgd'] = 2
        cfgs['log_heavy_interval'] = 1
        configs_outdir.save_configs(cfgs)

        out, err = run_command(
            f"drgnai train {configs_outdir.basepath} --load ")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
        assert 'Finished in ' in out, err

        assert (configs_outdir.outpath / 'weights.5.pkl').exists()
        assert not (configs_outdir.outpath / 'weights.6.pkl').exists()
        assert (configs_outdir.outpath / 'analysis_5').exists()
        assert configs_outdir.output_hash(5) == 5.689


@pytest.mark.parametrize(
    "configs_outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "hand.5.mrcs"),
                'ctf': os.path.join(DATA_DIR, "hand.5_ctf.pkl"),
                'n_imgs_pose_search': 10, 'epochs_sgd': 3,
                't_extent': 4.0, 't_n_grid': 2, 'z_dim': 4,
                'invert_data': False, 'seed': 7171, 'log_heavy_interval': 1,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'het'}})
        ],
    ids=("hand", ),
    indirect=True
    )
class TestHeterogeneousTrain:
    """Running heterogeneous reconstruction using command line arguments."""

    def test_reconstruction(self, configs_outdir):
        """Run training and any post-analyses."""
        out, err = run_command(f"drgnai train {configs_outdir.basepath}")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
        assert 'Finished in ' in out, err

        assert (configs_outdir.outpath / 'weights.5.pkl').exists()
        assert (configs_outdir.outpath / 'analysis_5').exists()
        assert configs_outdir.output_hash(5) == 8.354

    def test_separate_analyses(self, configs_outdir):
        """Run training and then check analyses for small experiments."""
        out, err = run_command(f"drgnai train {configs_outdir.basepath} --no-analysis")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
        assert 'Finished in ' in out, err

        assert (configs_outdir.outpath / 'weights.5.pkl').exists()
        assert not (configs_outdir.outpath / 'analysis_5').exists()
        assert configs_outdir.output_hash(5) == 8.354

        out, err = run_command(f"drgnai analyze {configs_outdir.basepath} "
                               "--ksample=2 --skip-umap")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err

        assert (configs_outdir.outpath / 'analysis_5' / 'z_pca.png').exists()
        assert configs_outdir.output_hash(5) == 8.354

        assert not (configs_outdir.outpath / 'analysis_3' / 'z_pca.png').exists()
        out, err = run_command(f"drgnai analyze {configs_outdir.basepath} "
                               "--epoch 3 --ksample=2 --skip-umap")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err

        assert (configs_outdir.outpath / 'analysis_3' / 'z_pca.png').exists()
        assert configs_outdir.output_hash(3) == 8.714

    def test_auto_analysis(self, configs_outdir):
        """Choosing an analysis epoch for an incomplete experiment."""
        out, err = run_command(f"drgnai train {configs_outdir.basepath} --no-analysis")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err

        assert configs_outdir.output_hash(5) == 8.354
        os.remove(os.path.join(configs_outdir.outpath, 'weights.5.pkl'))
        assert configs_outdir.output_hash(4) == 8.533

        out, err = run_command(f"drgnai analyze {configs_outdir.basepath} "
                               "--ksample=2 --skip-umap")
        assert (configs_outdir.outpath / 'analysis_4' / 'z_pca.png').exists(), err


@pytest.mark.parametrize(
    "configs_outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "toy_projections.mrcs"),
                'ctf': os.path.join(DATA_DIR, "test_ctf.pkl"),
                'pose': os.path.join(DATA_DIR, "toy_rot_trans.pkl"),
                'ind': 20, 'n_imgs_pose_search': 40, 'n_imgs_pretrain': 40,
                'epochs_sgd': 5, 't_extent': 4.0, 't_n_grid': 2, 'z_dim': zdim,
                'invert_data': False, 'seed': 307, 'log_heavy_interval': 5,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'fixed',
                                 'reconstruction_type': (
                                         'het' if zdim > 0 else 'homo')}})
        for zdim in (0, 4, 8)
        ],
    ids=("toy-homo", "toy-het4", "toy-het8"),
    indirect=True
    )
class TestFixedPoses:
    """Running reconstruction refinement with known poses."""

    def test_reconstruction(self, configs_outdir):
        """Run training and analyses."""
        out, err = run_command(f"drgnai train {configs_outdir.basepath} --no-analysis")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
        assert 'Finished in ' in out, err

        # confirm log_heavy_interval is working correctly
        assert (configs_outdir.outpath / 'weights.0.pkl').exists()
        assert not (configs_outdir.outpath / 'weights.4.pkl').exists()
        assert not (configs_outdir.outpath / 'weights.4.pkl').exists()
        assert (configs_outdir.outpath / 'weights.5.pkl').exists()

        zdim = configs_outdir.params['z_dim']
        assert configs_outdir.output_hash(5) == {0: 5.678, 4: -1.889, 8: 11.417}[zdim]


@pytest.mark.parametrize(
    "configs_outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "hand.5.mrcs"),
                'ctf': os.path.join(DATA_DIR, "hand.5_ctf.pkl"),
                'ind': os.path.join(DATA_DIR, "hand-ind3.pkl"),
                'n_imgs_pose_search': 6, 'epochs_sgd': 5, 'seed': 55,
                't_extent': 4.0, 't_n_grid': 2, 'log_heavy_interval': 1,
                'invert_data': False,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'homo'}})],
    ids=("hand-ind3", ),
    indirect=True
    )
class TestIndices:
    """Running homogeneous reconstruction with a index filter file."""

    def test_reconstruction(self, configs_outdir):
        """Run training and analyses."""
        out, err = run_command(f"drgnai train {configs_outdir.basepath} --no-analysis")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
        assert 'Finished in ' in out, err

        assert (configs_outdir.outpath / 'weights.7.pkl').exists()
        assert configs_outdir.output_hash(7) == 0.742

        out, err = run_command(f"drgnai analyze {configs_outdir.basepath} "
                               "--ksample=2 --skip-umap")
        assert (configs_outdir.outpath / 'analysis_7').exists(), err


@pytest.mark.xfail
@pytest.mark.parametrize(
    "configs_outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "M_particles_sub10.star"),
                'ctf': os.path.join(DATA_DIR, "M_ctf.pkl"),
                'datadir': os.path.join(DATA_DIR, "tilts"),
                'n_tilts': 1, 'dose_per_tilt': 2.93, 'angle_per_tilt': 3.0,
                'n_imgs_pose_search': 20, 'epochs_sgd': 3, 'seed': 2345,
                'invert_data': False, 'lazy': lazy_load,
                'quick_config': {'capture_setup': 'et',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'homo'}})
        for lazy_load in (True, False)
        ],
    ids=("lazy", "not-lazy"),
    indirect=True
    )
class TestTilts:
    """Running homogeneous reconstruction with a tilt series."""

    def test_reconstruction(self, configs_outdir):
        """Run training and analyses."""
        out, err = run_command(f"drgnai train {configs_outdir.basepath} --no-analysis")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
        assert 'Finished in ' in out, err

        assert (configs_outdir.outpath / 'weights.5.pkl').exists()
        assert configs_outdir.output_hash(5) == 2.309


@pytest.mark.xfail
@pytest.mark.parametrize(
    "configs_outdir", [
        dict(TEST_ARGS,
             **{'particles': os.path.join(DATA_DIR, "sta_testing_bin8.star"),
                'ctf': os.path.join(DATA_DIR, "sta_ctf.pkl"),
                'pose': os.path.join(DATA_DIR, "sta_pose.pkl"),
                'datadir': DATA_DIR, 'shuffle': shuffle, 'z_dim': 4,
                'log_heavy_interval': 1, 'epochs_sgd': 8, 'seed': 78,
                'n_tilts': 1, 'dose_per_tilt': 2.93, 'angle_per_tilt': 3.0,
                'use_gt_trans': True, 'n_imgs_pretrain': 40, 'lr': 0.015,
                'invert_data': False,
                'quick_config': {'capture_setup': 'et',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'refine',
                                 'reconstruction_type': 'het'}})
        for shuffle in (True, False)
        ],
    ids=("with-shuffling", "without-shuffling"),
    indirect=True
    )
class TestRefineTiltStar:
    """Running heterogeneous refinement using tilt series from a .star file."""

    def test_reconstruction(self, configs_outdir):
        """Run training and analyses."""
        out, err = run_command(f"drgnai train {configs_outdir.basepath} ")
        assert set(os.listdir(configs_outdir.basepath)) == {'out', 'configs.yaml'}, err
        assert 'Finished in ' in out, err

        assert (configs_outdir.outpath / 'weights.7.pkl').exists()
        assert round(configs_outdir.output_hash(7), 1) == {False: 213.5, True: 212.8}[
            configs_outdir.params['shuffle']]
