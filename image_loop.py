# Establish the input and output directories
in_directory = "/zpool/vladlab/active_drive/omaltz/git_repos/tensorflow_synthesis/original_teststim/*.PNG"
from glob import glob
import subprocess 
import tensorflow
image_paths = glob(in_directory)

# Establish the parameters 
POOLSIZE="1" # "1" for 1x1, "2" for 2x2, "3" for 3x3 or "4" for 4x4.
LAYER="pool5" # "pool1", "pool2", "pool3", "pool4", "pool5"
NUM_SAMPLES_TO_GENERATE=3

# Run tensorflow synthesis in loop
for im in image_paths:
    bash_cmd = f'python synthesize.py -i {im} -o /zpool/vladlab/active_drive/omaltz/git_repos/tensorflow_synthesis/output_teststim -p {POOLSIZE} -l {LAYER} -g 1 -s {NUM_SAMPLES_TO_GENERATE}'
    subprocess.run(bash_cmd, check = True, shell=True)