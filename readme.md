### READ THIS ###

-------------------------------TO SETUP ENV ----------------------
    
Run 

conda env create -p [SomePath]\.conda -f environment/environment.yaml

then conda activate [SomePath]\.conda

replacing [SomePath]

It should install everything needed. This assumes Cuda12.4. Change as needed - or remove pytorch gpu altogether if needed.  

Tensorboard is installed as well to monitor training.

to run:
tensorboard --logdir [SomePath]\runs


--------------------------TO RUN ----------------------

run main.py

Parameters can be chosen in hyp.yaml file. Currently setup to run 4x4 random grid, is_slippery = false, with the image
based agent. 

To change to run language-based agent, change type to 4 in hyp.yaml.

tensorboard files saved to /runs if want to inspect training curves. 
