"""Unit and fidelity tests of the pipeline command line interface."""

import os
import pytest
import yaml
from cryodrgnai.utils import run_command

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
DATASET_DIR = os.path.join(DATA_DIR, "data-paths.yaml")

with open(DATASET_DIR, 'r') as f:
    datasets = yaml.safe_load(f)


def test_test():
    """Test the test used to check if the package is installed correctly."""
    assert run_command("drgnai test") == ("Installation was successful!\n", "")


@pytest.mark.parametrize("dataset", list(datasets))
def test_no_outdir(dataset):
    """Test the interface without an output directory, which should error."""
    out, err = run_command(f"drgnai setup --dataset {dataset}")

    assert out == ""
    assert ("drgnai setup: error: the following "
            "arguments are required: outdir") in err


@pytest.mark.parametrize(
    "paths_file", ['hand-dataset.yaml', 'pose-dataset.yaml'])
@pytest.mark.parametrize("reconstruction_type", ["homo", "het", None])
@pytest.mark.parametrize(
    "pose_estimation", ["abinit", "refine", "fixed", None])
@pytest.mark.parametrize("conf_estimation", ["autodecoder", "encoder", None])
def test_file_dataset(outdir, paths_file,
                      reconstruction_type, pose_estimation, conf_estimation):
    """Test specifying a dataset from using a file containing input paths."""

    paths_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                              'data', 'hand-dataset.yaml')
    with open(paths_file, 'r') as f:
        paths = yaml.safe_load(f)

    if reconstruction_type:
        recon_arg = f"--reconstruction-type={reconstruction_type}"
    else:
        recon_arg = ""
        reconstruction_type = "homo"

    if pose_estimation:
        pose_arg = f"--pose-estimation={pose_estimation}"
    else:
        pose_arg = ""
        pose_estimation = "abinit"

    if conf_estimation:
        conf_arg = f"--conf-estimation={conf_estimation}"
    else:
        conf_arg = ""
        conf_estimation = "autodecoder"

    out, err = run_command(
        f"drgnai setup {outdir.basepath} --dataset {paths_file} "
        f"--capture-setup=spa {recon_arg} {pose_arg} {conf_arg}"
        )

    if "pose" in paths:
        use_pose = os.path.abspath(os.path.join(paths_file, paths['pose']))
        assert out == "" and err == ""
    else:
        use_pose = None

        if pose_estimation in {"refine", "fixed"}:
            assert "poses must be specified" in err
            assert not outdir.outpath.exists()
            return
        else:
            assert out == "" and err == ""

    assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
    assert os.path.isdir(os.path.join(outdir.basepath, "out"))

    if reconstruction_type == "homo":
        use_conf = None
    else:
        use_conf = conf_estimation

    assert outdir.configs == {
        'particles': os.path.abspath(
            os.path.join(paths_file, paths['particles'])),
        'ctf': os.path.abspath(os.path.join(paths_file, paths['ctf'])),
        'pose': use_pose, 'outdir': os.path.join(outdir.basepath, "out"),
        'quick_config': {'capture_setup': 'spa', 'conf_estimation': use_conf,
                         'pose_estimation': pose_estimation,
                         'reconstruction_type': reconstruction_type}
        }


@pytest.mark.parametrize("reconstruction_type", ["homo", "het", None])
@pytest.mark.parametrize(
    "pose_estimation", ["abinit", "refine", "fixed", None])
@pytest.mark.parametrize("conf_estimation", ["autodecoder", "encoder", None])
@pytest.mark.parametrize("dataset", list(datasets))
def test_label_dataset(outdir, dataset,
                       reconstruction_type, pose_estimation, conf_estimation):
    if "dose_per_tilt" in datasets[dataset]:
        capture_setup = "et"
    else:
        capture_setup = "spa"

    if reconstruction_type:
        recon_arg = f"--reconstruction-type={reconstruction_type}"
    else:
        recon_arg = ""
        reconstruction_type = "homo"

    if pose_estimation:
        pose_arg = f"--pose-estimation={pose_estimation}"
    else:
        pose_arg = ""
        pose_estimation = "abinit"

    if conf_estimation:
        conf_arg = f"--conf-estimation={conf_estimation}"
    else:
        conf_arg = ""
        conf_estimation = "autodecoder"

    out, err = run_command(
        f"drgnai setup {outdir.basepath} --dataset {dataset} "
        f"--capture-setup={capture_setup} {recon_arg} {pose_arg} {conf_arg} "
        )

    if capture_setup == "et":
        if reconstruction_type == "het" and conf_estimation == "encoder":
            assert ("encoder is not implemented "
                    "for subtomogram averaging") in err
            assert not outdir.outpath.exists()
            return

    if "pose" in datasets[dataset]:
        use_pose = datasets[dataset]['pose']
        assert out == "" and err == ""
    else:
        use_pose = None

        if pose_estimation in {"refine", "fixed"}:
            assert "poses must be specified" in err
            assert not outdir.outpath.exists()
            return
        else:
            assert out == "" and err == ""

    assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
    assert os.path.isdir(os.path.join(outdir.basepath, "out"))

    if reconstruction_type == "homo":
        use_conf = None
    else:
        use_conf = conf_estimation

    check_configs = {
        'particles': os.path.abspath(datasets[dataset]['particles']),
        'ctf': os.path.abspath(datasets[dataset]['ctf']),
        'pose': os.path.abspath(use_pose) if use_pose else None,
        'dataset': dataset, 'outdir': os.path.join(outdir.basepath, "out"),
        'quick_config': {'capture_setup': capture_setup,
                         'conf_estimation': use_conf,
                         'pose_estimation': pose_estimation,
                         'reconstruction_type': reconstruction_type}
        }

    if capture_setup == "et":
        check_configs['dose_per_tilt'] = datasets[dataset]['dose_per_tilt']

    assert outdir.configs == check_configs


@pytest.mark.parametrize("reconstruction_type", ["homo", "het", None])
@pytest.mark.parametrize(
    "pose_estimation", ["abinit", "refine", "fixed", None])
@pytest.mark.parametrize("conf_estimation", ["autodecoder", "encoder", None])
@pytest.mark.parametrize("dataset", list(datasets))
def test_explicit_dataset(outdir, dataset, reconstruction_type,
                          pose_estimation, conf_estimation):
    if reconstruction_type:
        recon_arg = f"--reconstruction-type={reconstruction_type}"
    else:
        recon_arg = ""
        reconstruction_type = "homo"

    if pose_estimation:
        pose_arg = f"--pose-estimation={pose_estimation}"
    else:
        pose_arg = ""
        pose_estimation = "abinit"

    if conf_estimation:
        conf_arg = f"--conf-estimation={conf_estimation}"
    else:
        conf_arg = ""
        conf_estimation = "autodecoder"

    if "pose" in datasets[dataset]:
        poses_arg = f"--pose={datasets[dataset]['pose']}"
    else:
        poses_arg = ""

    out, err = run_command(f"drgnai setup {outdir.basepath} "
                           f"--particles {datasets[dataset]['particles']} "
                           f"--ctf {datasets[dataset]['ctf']} "
                           f"{poses_arg} {recon_arg} {pose_arg} {conf_arg}")

    if "pose" in datasets[dataset]:
        use_pose = datasets[dataset]['pose']
        assert out == "" and err == ""
    else:
        use_pose = None

        if pose_estimation in {"refine", "fixed"}:
            assert "poses must be specified" in err
            assert not outdir.outpath.exists()
            return
        else:
            assert out == "" and err == ""

    assert set(os.listdir(outdir.basepath)) == {'out', 'configs.yaml'}
    assert os.path.isdir(os.path.join(outdir.basepath, "out"))

    if reconstruction_type == "homo":
        use_conf = None
    else:
        use_conf = conf_estimation

    assert outdir.configs == {
        'particles': os.path.abspath(datasets[dataset]['particles']),
        'ctf': os.path.abspath(datasets[dataset]['ctf']),
        'pose': os.path.abspath(use_pose) if use_pose else None,
        'outdir': os.path.join(outdir.basepath, "out"),
        'quick_config': {'capture_setup': 'spa', 'conf_estimation': use_conf,
                         'pose_estimation': pose_estimation,
                         'reconstruction_type': reconstruction_type}
        }
