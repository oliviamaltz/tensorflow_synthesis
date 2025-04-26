#!/bin/bash
scaling=0.3

for image in rocks zebras grass face tulips elephant; 
do 
  for layer in pool1 pool2 pool4; 
  do 
    sbatch -p hns,gpu --gres gpu:1 --mem=10G --time=02:00:00 --wrap="module load py-scikit-image/0.15.0_py27; module load py-pytorch/1.0.0_py27; module load py-scikit-learn; module load py-scipy/1.1.0_py27; module load py-numpy/1.14.3_py27; module load py-matplotlib/2.2.2_py27; module load py-tensorflow/1.6.0_py27; cd ~/TextureSynthesis/tensorflow_synthesis; python runSpatial.py -i $image -f ${image}-RF_${scaling}-${layer} -l $layer -g 1 -s 3 -p $scaling" 
  done 
done

