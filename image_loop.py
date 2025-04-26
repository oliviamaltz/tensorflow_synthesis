# Establish the input and output directories
in_directory = "/home/arsch/image_scrambler/*/*.JPEG"
from glob import glob
import subprocess 
import tensorflow
image_paths = glob(in_directory)

# Establish the parameters 
POOLSIZE="1" # "1" for 1x1, "2" for 2x2, "3" for 3x3 or "4" for 4x4.
LAYER="pool4" # "pool1", "pool2", "pool3", "pool4", "pool5"
NUM_SAMPLES_TO_GENERATE=1

# Run tensorflow synthesis in loop
for im in image_paths:
    bash_cmd = f'python synthesize.py -i {im} -o /home/arsch/image_scrambler/output -p {POOLSIZE} -l {LAYER} -g 1 -s {NUM_SAMPLES_TO_GENERATE}'
    subprocess.run(bash_cmd, check = True, shell=True)