# :dragon::robot: CryoDRGN-AI: _Ab initio_ cryo-EM reconstruction #

CryoDRGN-AI is a neural network-based algorithm for _ab initio_ heterogeneous cryo-EM reconstruction.
The method leverages the expressive representation capacity of neural models and implements a
two-stage joint inference procedure of poses and heterogeneous conformational states to enable single-shot
reconstruction of noisy, large cryo-EM datasets. 

Note: We are currently merging this standalone version of cryoDRGN-AI into the [cryodrgn](https://github.com/ml-struct-bio/cryodrgn) software package. Please check back for updates. 

## Documentation ##

The latest detailed documentation for CryoDRGN-AI is available [on gitbook](https://ez-lab.gitbook.io/drgn-ai/), 
including an overview and walkthrough of CryoDRGN-AI installation, training and analysis. A brief quick start is
provided below.


## New in Version 0.3.2-beta ##

 - Add `invert_data` configuration parameter for easier handling of datasets like 50S (EMPIAR-10076)
 - Add `data_norm_mean` and `data_norm_std` configuration parameters for manual override of image dataset normalization
 - Updating default behavior to use only one GPU no matter how many are available; add `--multigpu` to `drgnai train`
   as well as `multigpu` configuration parameter to manually activate multi-GPU training
 - Using volumes reconstructed at the image closest to the centroid of all images in the latent space at each epoch
   instead of the first image in the dataset for heterogeneous reconstruction
 - Fixing loss logging during pretraining; better logs for how many GPUs are being used


### New in Version 0.3.1-beta ###

 - Add `--datadir` to `drgnai setup`
 - Remove `outdir` from the `config.yaml`
 - Allow arbitrary config parameters to be passed to `drgnai setup`
 - Add `--load` in `drgnai train` for auto-restart of experiments
 - Always save last epoch
 - Renaming of old experiments in the same output folder as `old-out_000_fixed-homo/`, `old-out_001_abinit-het4/`, `...`
   instead of just `out_old/`
 - Fixing pose search [bug](https://github.com/ml-struct-bio/drgnai-internal/issues/123) found in previous version


## Installation ##

We recommend installing CryoDRGN-AI in a clean conda environment — first clone the latest stable version available in 
the git repository, and then use `pip` to install the package from the source code:

    (base) $ conda create --name drgnai python=3.9
    (base) $ conda activate drgnai
    (drgnai) $ git clone git@github.com:ml-struct-bio/drgnai.git
    (drgnai) $ cd drgnai/
    (drgnai) $ pip install . 

To confirm that the package was installed successfully, use `drgnai test`:

```
(drgnai) $ drgnai test
Installation was successful!
```

You may also choose to define an environment variable `$DRGNAI_DATASETS` in your bash environment, which will allow you
to point to a file listing locations of input files and dataset labels to use as shortcuts. For more information, 
see our [detailed user guide](https://ez-lab.gitbook.io/drgn-ai/).


## Usage ##

This package installs the `drgnai` command line tool for running experiments, which contains three key subcommands:

 - `drgnai setup` creates the experiment folder and configuration parameter file
 - `drgnai train` trains and analyzes a reconstruction model
 - `drgnai analyze` performs specific analyses in addition to those done by `train`

As cryo-EM reconstruction experiments are usually computationally intensive, `train` especially is most
commonly used within a script submitted to a job scheduling system on a high-performance compute cluster.


### Setup ###

First, use the `drgnai setup` tool to create an output directory and a configuration file for CryoDRGN-AI. 

```
drgnai setup your_outdir --particles /my_data/particles.mrcs --ctf /my_data/ctf.pkl \
                         --capture-setup spa --conf-estimation autodecoder \
                         --pose-estimation abinit --reconstruction-type het                               
```

This command will create an output directory called `your_outdir/` and a configuration file `your_outdir/configs.yaml`:

```yaml
particles: /my_data/particles.mrcs
ctf: /my_data/ctf.pkl
quick_config:
  capture_setup: spa
  conf_estimation: autodecoder
  pose_estimation: abinit
  reconstruction_type: het
```

Other configurations parameters can then still be added to `configs.yaml`;
see [our documentation](https://ez-lab.gitbook.io/drgn-ai/full-documentation) for more details.
We especially recommend `lazy: true` to avoid memory issues with larger datasets, as well `lr` to control the 
learning rate of the reconstruction model.


### Reconstruction and analysis ###

After setup is complete, run the experiment using `drgnai train your_outdir/`. 

```
drgnai train your_outdir/
```

`drgnai` will save the outputs of training under `your_outdir/out/`.
By default, at the end of training, `drgnai` will analyze the results from the last epoch. 


You can also run analyses on a particular training epoch instead of the last epoch.
Outputs of each analysis will be stored under `your_outdir/out/analysis_<epoch>/`.

```
drgnai analyze your_outdir/ --epoch 25
```


### Monitoring running experiments ###

The progress of model training can be tracked using the `your_outdir/out/training.log` file.

The training step can also be monitored while it is running using Tensorboard, which is installed as part of DRGN-AI,
by following these steps:

1. Run the command `tensorboard --logdir out-dir/out --port 6565 --bind_all` remotely, where out-dir is the experiment 
output directory and 6565 is an arbitrary port number.
2. Run the command `ssh -NfL 6565:<server-name>:6565 <user-name>@<server-address>` locally, using the same port number 
   above, and replacing the server info with your own.
3. Navigate to localhost:6565 in your local browser to access the tensorboard interface.


## Advanced Configuration ##

The behavior of the algorithm can be modified by passing different values to `drgnai setup` at the beginning of the
experiment. However, only the most important parameters are available through this interface:

 - `--capture-setup` “spa” for single-particle analysis (default, only mode currently available)
 - `--reconstruction-type` “het” for heterogeneous or “homo” for homogeneous (default)
 - `--pose-estimation` “abinit” for no initialization (default), “refine” to refine provided poses by gradient
                       descent or “fixed” to use provided poses without refinement
 - `--conf-estimation` “autodecoder” (default), “encoder” or “refine” to refine conformations by
                       gradient descent (you must then define initial_conf) — not used in homogeneous reconstruction

Note that each argument can be specified using a non-ambiguous prefix as a shortcut 😃, e.g.
```
drgnai setup out-dir --dataset 50S_128 --cap spa --conf autodecoder --pose-estim abinit --reconstr het
```

To change the other configuration parameters, the `configs.yaml` file must be edited directly before the experiment
is run. For a full overview of how to configure the parameters used in the CryoDRGN-AI model, see the
[docs](https://ez-lab.gitbook.io/drgn-ai/configuration).


## Reference ##

```
@article{cryodrgnai,
  title = "CryoDRGN-AI: neural ab initio reconstruction of challenging cryo-EM and cryo-ET datasets",
  author = "Levy, Axel and Raghu, Rishwanth and Feathers, Ryan and Grzadkowski, Michal and Poitevin, Frederic and
            Johnston, Jake D, Vallese, Francesca and Clarke, Oliver B and Wetzstein, Gordon and Zhong, Ellen D",
  journal = "Nature Methods",
  doi = 10.1038/s41592-025-02720-4,
  url = nature.com/articles/s41592-025-02720-4,
  month = jun,
  year = 2025,
}
```


## Previous versions ##

### Version 0.2.2-beta ###

 - `drgnai filter` interface for interactive filtering of particles
 - Support for `$DRGNAI_DATASETS` dataset catalogue
 - Cleaner tracking of configuration specifications

### Version 0.2.0-beta ###

 - Creating a new `drgnai` command line interface for easier use of the package
 - Simplifying how configuration files are used
 - New version of the README, creating gitbook documentation


## Contact ##

For any feedback, questions, or bugs, please file a Github issue or reach out to the authors.
