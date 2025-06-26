### A smoke test for ab-initio reconstruction using the hand dataset

set -e
set -x

# create experiment output folder and configuration file
drgnai setup --particles data/hand.mrcs --ctf data/hand_ctf.pkl \
             --reconstruction-type homo --pose-estimation abinit hand-test

# manipulate the config gile to create an "easy" model
mv hand-test/configs.yaml hand-test/configs_base.yaml
cat hand-test/configs_base.yaml data/hand_model.yaml > hand-test/configs.yaml

drgnai train hand-test
cryodrgn_utils invert_contrast hand-test/out/reconstruct.7.mrc
