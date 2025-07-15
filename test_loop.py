import subprocess
from pathlib import Path

# Input and output directories
input_dir = Path("/zpool/vladlab/active_drive/omaltz/git_repos/tensorflow_synthesis/original_teststim")
output_dir = "/zpool/vladlab/active_drive/omaltz/git_repos/tensorflow_synthesis/output_teststim"

# Synthesis parameters
poolsize = "1"
layer = "pool5"
num_samples = "3"

# Gather all image files
image_paths = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

# Run synthesize.py for each image
for im_path in image_paths:
    cmd = f'python synthesize.py -i "{im_path}" -o "{output_dir}" -p {poolsize} -l {layer} -g 1 -s {num_samples}'
    print(f"🚀 Running: {cmd}")
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"Finished: {im_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {im_path.name}")
        print(e)
