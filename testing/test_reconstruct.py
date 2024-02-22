"""Unit and fidelity tests of the pipeline command line interface."""

import os
import pytest
import yaml
from cryodrgnai.utils import run_command

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
DATASET_DIR = os.path.join(DATA_DIR, "data-paths.yaml")


@pytest.mark.parametrize(
    "outdir", [{'particles': os.path.join(DATA_DIR, "toy_projections.mrcs"),
                'ctf': os.path.join(DATA_DIR, "test_ctf.pkl"),
                'ind': 15, 'n_imgs_pose_search': 30, 'epochs_sgd': 8,
                'z_dim': 4, 'seed': 771,
                'hypervolume_dim': 32, 'hypervolume_layers': 2,
                'log_heavy_interval': 2,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'het'}}],
    ids=("toy", ),
    indirect=True
    )
def test_small_dataset_and_load(outdir):
    """Create a toy dataset using subset indices and run the pipeline."""
    out, err = run_command(f"drgnai train {outdir.basepath}")

    assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
    assert 'Finished in ' in out
    assert (outdir.outpath / 'weights.8.pkl').exists()
    assert not (outdir.outpath / 'analysis_8').exists()
    assert (outdir.outpath / 'weights.10.pkl').exists()
    assert (outdir.outpath / 'analysis_10').exists()

    assert outdir.hvolm_hash(8) == -0.744
    assert outdir.conf_hash(8) == 0.210
    assert outdir.hvolm_hash(10) == -0.456
    assert outdir.conf_hash(10) == 0.176

    new_configs = outdir.configs
    new_configs['seed'] = 901
    new_configs['load'] = os.path.abspath(
        os.path.join(outdir.basepath, 'out_old', 'weights.8.pkl'))
    with open(os.path.join(outdir.basepath, "configs.yaml"), 'w') as f:
        yaml.dump(new_configs, f)

    out, err = run_command(f"drgnai train {outdir.basepath}")

    assert set(os.listdir(outdir.basepath)) == {
        'out_old', 'out', 'configs.yaml'}
    assert 'Finished in ' in out
    assert not (outdir.outpath / 'analysis_16').exists()
    assert (outdir.outpath / 'analysis_18').exists()
    for i in range(10, 20, 2):
        assert (outdir.outpath / f'weights.{i}.pkl').exists()

    assert outdir.hvolm_hash(16) == 0.676
    assert outdir.conf_hash(16) == 0.019
    assert outdir.hvolm_hash(18) == 1.132
    assert outdir.conf_hash(18) == -0.02


@pytest.mark.parametrize(
    "outdir", [{'particles': os.path.join(DATA_DIR, "hand.5.mrcs"),
                'ctf': os.path.join(DATA_DIR, "ctf1.pkl"),
                'n_imgs_pose_search': 10, 'epochs_sgd': 5,
                'z_dim': 4, 'seed': 1331,
                'hypervolume_dim': 16, 'hypervolume_layers': 2,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'homo'}}],
    ids=("hand", ),
    indirect=True
    )
class TestHomogeneousTrain:
    """Running homogeneous reconstruction using command line arguments."""

    def test_reconstruction(self, outdir):
        out, err = run_command(f"drgnai train {outdir.basepath}")

        assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
        assert 'Finished in ' in out
        assert (outdir.outpath / 'weights.5.pkl').exists()
        assert (outdir.outpath / 'analysis_5').exists()
        assert outdir.hvolm_hash(5) == -2.260
        assert outdir.conf_hash(5) == 0.814

    def test_just_train(self, outdir):
        out, err = run_command(f"drgnai train {outdir.basepath} --no-analysis")

        assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
        assert 'Finished in ' in out
        assert (outdir.outpath / 'weights.5.pkl').exists()
        assert not (outdir.outpath / 'analysis_5').exists()
        assert outdir.hvolm_hash(5) == -2.260
        assert outdir.conf_hash(5) == 0.814


@pytest.mark.parametrize(
    "outdir", [{'particles': os.path.join(DATA_DIR, "hand.5.mrcs"),
                'ctf': os.path.join(DATA_DIR, "ctf1.pkl"),
                'n_imgs_pose_search': 10, 'epochs_sgd': 5,
                'z_dim': 4, 'seed': 7171,
                'hypervolume_dim': 16, 'hypervolume_layers': 2,
                'log_heavy_interval': 1, 'num_workers': 1,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'het'}}],
    ids=("hand", ),
    indirect=True
    )
class TestHeterogeneousTrain:
    """Running heterogeneous reconstruction using command line arguments."""

    def test_reconstruction(self, outdir):
        """Run training and any post-analyses."""
        out, err = run_command(f"drgnai train {outdir.basepath}")

        assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
        assert 'Finished in ' in out
        assert err == ""
        assert (outdir.outpath / 'weights.7.pkl').exists()
        assert (outdir.outpath / 'analysis_7').exists()
        assert outdir.hvolm_hash(5) == -3.350
        assert outdir.conf_hash(5) == 0.453

    def test_separate_analyses(self, outdir):
        """Run training and then check analyses for small experiments."""
        out, err = run_command(f"drgnai train {outdir.basepath} --no-analysis")

        assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
        assert 'Finished in ' in out
        assert err == ""
        assert (outdir.outpath / 'weights.7.pkl').exists()
        assert not (outdir.outpath / 'analysis_7').exists()
        assert outdir.hvolm_hash(5) == -3.350
        assert outdir.conf_hash(5) == 0.453

        out, err = run_command(f"drgnai analyze {outdir.basepath} "
                               "--ksample=2 --skip-umap")

        assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
        assert err == ""
        assert (outdir.outpath / 'analysis_7' / 'z_pca.png').exists()
        assert outdir.hvolm_hash(5) == -3.350
        assert outdir.conf_hash(5) == 0.453

        out, err = run_command(f"drgnai analyze {outdir.basepath} "
                               "--epoch 3 --ksample=2 --skip-umap")

        assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
        assert err == ""
        assert (outdir.outpath / 'analysis_3' / 'z_pca.png').exists()
        assert outdir.hvolm_hash(3) == -3.374
        assert outdir.conf_hash(3) == 0.255

    def test_auto_analysis(self, outdir):
        """Choosing an analysis epoch for an incomplete experiment."""
        out, err = run_command(f"drgnai train {outdir.basepath} --no-analysis")

        assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
        assert err == ""
        assert outdir.hvolm_hash(4) == -3.363
        assert outdir.conf_hash(4) == 0.354

        os.remove(os.path.join(outdir.outpath, 'weights.7.pkl'))
        out, err = run_command(f"drgnai analyze {outdir.basepath} "
                               "--ksample=2 --skip-umap")

        assert err == ""
        assert (outdir.outpath / 'analysis_6' / 'z_pca.png').exists()


@pytest.mark.parametrize(
    "outdir", [{'particles': os.path.join(DATA_DIR, "hand.5.mrcs"),
                'ctf': os.path.join(DATA_DIR, "ctf1.pkl"),
                'ind': os.path.join(DATA_DIR, "hand-ind3.pkl"),
                'n_imgs_pose_search': 6, 'epochs_sgd': 5, 'seed': 55,
                'hypervolume_dim': 16, 'hypervolume_layers': 2,
                'log_heavy_interval': 1, 'num_workers': 1,
                'quick_config': {'capture_setup': 'spa',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'homo'}}],
    ids=("hand-ind3", ),
    indirect=True
    )
class TestIndices:
    """Running homogeneous reconstruction with a index filter file."""

    def test_reconstruction(self, outdir):
        """Run training and analyses."""
        out, err = run_command(f"drgnai train {outdir.basepath} --no-analysis")

        assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
        assert 'Finished in ' in out
        assert err == ""
        assert (outdir.outpath / 'weights.7.pkl').exists()
        assert outdir.hvolm_hash(5) == -3.315

        out, err = run_command(f"drgnai analyze {outdir.basepath} "
                               "--ksample=2 --skip-umap")
        assert err == ""
        assert (outdir.outpath / 'analysis_7').exists()


@pytest.mark.parametrize(
    "outdir", [{'particles': os.path.join(DATA_DIR, "M_particles_sub10.star"),
                'ctf': os.path.join(DATA_DIR, "M_ctf.pkl"),
                'datadir': os.path.join(DATA_DIR, "tilts"),
                'n_tilts': 1, 'n_tilts_pose_search': 1,
                'n_imgs_pose_search': 20, 'epochs_sgd': 3, 'seed': 77,
                'hypervolume_dim': 16, 'hypervolume_layers': 2,
                'log_heavy_interval': 1, 'num_workers': 1,
                'quick_config': {'capture_setup': 'et',
                                 'conf_estimation': 'autodecoder',
                                 'pose_estimation': 'abinit',
                                 'reconstruction_type': 'homo'}}],
    indirect=True
    )
class TestTilts:
    """Running homogeneous reconstruction with a tilt series."""

    def test_reconstruction(self, outdir):
        """Run training and analyses."""
        out, err = run_command(f"drgnai train {outdir.basepath} --no-analysis")

        assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
        assert 'Finished in ' in out
        assert (outdir.outpath / 'weights.5.pkl').exists()
        assert outdir.hvolm_hash(5) == -1.919
