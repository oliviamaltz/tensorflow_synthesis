orig_im_dir="$PI_SCRATCH/texture_stimuli/color/new_originals"
save_dir="$PI_SCRATCH/texture_stimuli/color/new_textures"

for img in apple bear cat cruiseship dalmatian ferrari greatdane helicopter horse house iphone jordan laptop quarterback samosa shoes stephcurry tiger truck; 
do 
  for layer in "pool1" "pool2" "pool4"; 
  do 
    sbatch -p hns,gpu --gres gpu:1 --mem=5G --time=03:00:00 --wrap="module load py-scikit-image/0.15.0_py27; module load py-pytorch/1.0.0_py27; module load py-scikit-learn; module load py-scipy/1.1.0_py27; module load py-numpy/1.14.3_py27; module load py-matplotlib/2.2.2_py27; module load py-tensorflow/1.6.0_py27; cd $HOME/TextureSynthesis/tensorflow_synthesis; python synthesize.py -d $orig_im_dir -i $img -o $save_dir -g 1 -s 4 -p 1 -l $layer; python synthesize.py -d $orig_im_dir -o $save_dir -i $img -g 1 -s 4 -p 2 -l $layer; python synthesize.py -d $orig_im_dir -o $save_dir -i $img -g 1 -s 4 -p 3 -l $layer; python synthesize.py -d $orig_im_dir -o $save_dir -i $img -g 1 -s 4 -p 4 -l $layer;"; 
  sleep 0.2; 
done done
