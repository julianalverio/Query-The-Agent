conda create -n new_her python=3.9
conda activate new_her
conda install pytorch torchvision  cudatoolkit=11.1 -c pytorch -c nvidia
pip install mujoco_py==2.0.2.4 tensorboard pyyaml sklearn
