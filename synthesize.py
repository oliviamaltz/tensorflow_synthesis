import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import sys, time
import argparse

import TextureSynthesis as ts
#from VGGWeights import *
from VGG19 import *
from skimage.io import imread, imsave
from skimage.transform import resize

import pickle

def main(args):
    # Keep track of how long this all takes
    start_time = time.time()

    # Load VGG-19 weights and build model
    weights_file = 'vgg19_normalized.pkl'
    
    # Load VGG-19 weights and build model.
    with open(weights_file, 'rb') as f:
        vgg_weights = pickle.load(f, encoding='latin1')['param values']
    vgg19 = VGG19(vgg_weights)
    vgg19.build_model()

    # Weights for each layer
    all_layers = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5']
    assert args.layer in all_layers, 'Specified layer must be in {}'.format(all_layers)
    layer_weight = {x: 1e9 for x in all_layers[:all_layers.index(args.layer)+1]}

    # Load up original image
    image_path = args.imagepath #'{}/{}.jpg'.format(args.inputdir, args.image)
    args.image = image_path.split('/')[-1].split('.')[0]
    original_image = preprocess_im(image_path)

    # Make a temporary directory to save the intermediates in.
    tmpDir = "%s/iters" % (args.outputdir)
    os.system("mkdir -p %s" %(tmpDir))

    print("Synthesizing texture", args.image, "matching model", args.layer, "for", args.iterations, "iterations, with nPools:", args.nPools)

    # Initialize texture synthesis
    texsyn = ts.TextureSynthesis(vgg19, original_image, layer_weight, args.nPools, args.layer, args.image, tmpDir, args.iterations+1)

    # Do training
    if args.generateMultiple==1:
        for i in range(args.sampleidx):
            print('Generating sample {} of {}'.format(i+1, args.sampleidx))
            texsyn.train(i+1, loss=args.loss)
    else:
        texsyn.train(args.sampleidx, loss=args.loss) 
    postprocess_img(tmpDir, args)

    print('Done synthesizing all images. Took {:.2f} seconds in total.'.format(time.time()-start_time))
    sys.stdout.flush()

    return texsyn

def preprocess_im(path):
    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    image = imread(path)

    if image.shape[1]!=256 or image.shape[0]!=256:
        image = resize(image, (256,256), preserve_range=True, anti_aliasing=True).astype('uint8')

    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    if len(image.shape)<4:
        image = np.stack((image,image,image),axis=3)

    # If there is a Alpha channel, just scrap it
    if image.shape[3] == 4:
        image = image[:,:,:,:3]

    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image

def postprocess_img(raw, args, steps=None):
    for im in os.listdir(raw):
        if '{}_{}x{}_{}_smp'.format(args.image, args.nPools, args.nPools, args.layer) in im  and ('final' in im or 'step_{}.npy'.format(args.iterations) in im or 'step_{}.npy'.format(steps) in im): 
            path_to_img = raw+'/'+im
            filename = im[:im.index('_step_')]
            if steps is not None and steps!=args.iterations:
              filename += '_step{}'.format(steps)
            if 'final' in im:
              filename += '_final'
            save_path = '{}/{}'.format(args.outputdir, filename)
            if not os.path.isdir('{}/npy'.format(args.outputdir)):
              os.mkdir('{}/npy'.format(args.outputdir))

            if filename + '.png' not in os.listdir(args.outputdir):
              image = np.load(path_to_img)

              # Save as PNG into outdir
              imsave(save_path + '.png', image)

              # Also copy .npy fileinto outdir
              os.system('cp {} {}/npy/{}.npy'.format(path_to_img, args.outputdir, filename))
              print(im + ' saved as PNG to ' + args.outputdir)


def test_model(args):
    # Load VGG-19 weights and build model
    weights_file = 'vgg19_normalized.pkl'
    
    # Load VGG-19 weights and build model.
    with open(weights_file, 'rb') as f:
        vgg_weights = pickle.load(f, encoding='latin1')['param values']
    vgg19 = VGG19(vgg_weights)
    vgg19.build_model()

    # Weights for each layer
    all_layers = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5']
    assert args.layer in all_layers, 'Specified layer must be in {}'.format(all_layers)
    layer_weight = {x: 1e9 for x in all_layers[:all_layers.index(args.layer)+1]}

    # Load up original image
    image_path = args.imagepath 
    original_image = preprocess_im(image_path)

    # Make a temporary directory to save the intermediates in.
    tmpDir = "%s/iters" % (args.outputdir)
    os.system("mkdir -p %s" %(tmpDir))

    print("Synthesizing texture", args.image, "matching model", args.layer, "for", args.iterations, "iterations, with nPools:", args.nPools)

    # Initialize texture synthesis
    texsyn = ts.TextureSynthesis(vgg19, original_image, layer_weight, args.nPools, args.layer, args.image, tmpDir, args.iterations+1)
    return texsyn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layer", default="pool4")
    # parser.add_argument("-d", "--inputdir", default="/home/arsch/image_scrambler/original")
    parser.add_argument("-o", "--outputdir", default="/home/arsch/image_scrambler/output")
    parser.add_argument("-i", "--imagepath", default="/home/arsch/image_scrambler/original")
    parser.add_argument("-s", "--sampleidx", type=int, default=1)
    parser.add_argument("-p", "--nPools", type=int, default=1)
    parser.add_argument('-g', '--generateMultiple',type=int, default=0)
    parser.add_argument('-n', '--iterations', type=int, default=5000)
    parser.add_argument('-k', '--loss', default='texture') # "texture" (gramian only) or "spectral" (gramian + FFT)
    args = parser.parse_args()
    
    if not os.path.isfile(args.imagepath):
        print(f'Error: image {args.imagepath} not found. Quitting...')
    else:
        texsyn = main(args)

    #texsyn = test_model(args)
    #raw = args.outputdir + '/iters'
    #postprocess_img(raw, args)
 
