conda create -n baselines python=3.6
conda activate baselines
conda install patchelf
pip install tensorflow-gpu==1.15 pyyaml scipy sklearn tabulate==0.8.7 dill==0.2.9 mujoco_py==2.0.2.4 matplotlib==3.1.3 opencv-python tensorboard_logger mpi4py cloudpickle
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

