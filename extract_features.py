from TextureSynthesis import *
import numpy as np
import tensorflow as tf
import sys, os, pickle, argparse
from tqdm import tqdm
from synthesize import preprocess_im
import pdb

class VGG19_FeatureExtractor:
    def __init__(self, feature_type='gramian', nPools=1, inputdir='/scratch/groups/jlg/texture_stimuli/color', outdir = '/scratch/groups/jlg/texture_stimuli/color/deepnet_features2'):
        self.inputdir = inputdir
        self.outdir = outdir
        self.nPools=nPools
        self.feature_type = feature_type

        # Load VGG-19 weights and build model.
        weights_file = 'vgg19_normalized.pkl'
        with open(weights_file, 'rb') as f:
            vgg_weights = pickle.load(f)['param values']
        vgg19 = VGG19(vgg_weights)
        vgg19.build_model()

        # Define layers
        self.layer_names = ['conv1_1', 'conv1_2','pool1','conv2_1', 'conv2_2', 'pool2', 'conv3_1', \
                               'conv3_2','pool3','conv4_1','conv4_2','conv4_3','conv4_4','pool4',\
                               'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5']
        layer_weights = {layer: 1e9 for layer in self.layer_names}

        # set some arbitrary image as the initial input, we'll anyway change this later
        original_image = np.load('{}/originals/rocks.npy'.format(self.inputdir)) 

        # Create the TextureSynthesis object which will compute the weighted Gramian, etc.
        self.texsynth = TextureSynthesis(vgg19, original_image, layer_weights, nPools)

        # Load the list of images to get features for.
        self.filenames = np.load('{}/CORnet-Z_filenames.npy'.format(self.outdir))
        np.save('{}/VGG19_filenames.npy'.format(self.outdir), self.filenames)

        if self.feature_type == 'gramian':
            print('Getting {}x{} gramians for {} images'.format(nPools, nPools, len(self.filenames)))
            self.get_gramians()
        elif self.feature_type == 'diagonal':
            print('Getting Diagonal of Gramian for {} images'.format(len(self.filenames)))
            self.get_gramians(get_diag=True)
        else:
            print('Getting activation features for {} images'.format(len(self.filenames)))
            self.get_activations()


    def get_gramians(self, get_diag=False):
        # Loop through all the files and extract features.
        all_features = {layer: [] for layer in self.layer_names}
        for fi, filename in tqdm(enumerate(self.filenames), total=len(self.filenames)):
            # Load the numpy file corresponding to each image
            #npy_file = filename.split('.')[0] + '.npy'
            #image = np.load(npy_file)
            image = preprocess_im(filename)
            if len(image.shape)==3: image = np.expand_dims(image, 0)

            # Get the gramian from the texturesynthesis object
            gram = self.texsynth._get_constraints(original_image=image)

            for layer in gram:
                if get_diag:
                    all_features[layer].append(np.diag(gram[layer].squeeze()))
                else:
                    all_features[layer].append(gram[layer])

        for layer in all_features:
            features = np.array(all_features[layer]).reshape(len(self.filenames),-1)
            print(layer, features.shape)
            if get_diag:
                savename = 'VGG19_{}_output_diagonal.npy'.format(layer)
            else:
                savename = 'VGG19_{}_output_{}x{}gramian.npy'.format(layer, nPools, nPools)
            np.save('{}/{}'.format(self.outdir, savename), features) 

    def get_activations(self):
        # Loop through all the files and extract features.
        all_features = {layer: [] for layer in self.layer_names}
        for fi, filename in tqdm(enumerate(self.filenames), total=len(self.filenames)):

            #npy_file = filename.split('.')[0] + '.npy'
            #image = np.load(npy_file)
            image = preprocess_im(filename)
            if len(image.shape)==3: image = np.expand_dims(image, 0)
            gram = self.texsynth._get_activations(original_image=image)

            for layer in gram:
                all_features[layer].append(gram[layer])

        for layer in all_features:
            features = np.array(all_features[layer]).reshape(len(self.filenames),-1)
            print(layer, features.shape)
            np.save('{}/VGG19_{}_output_feats.npy'.format(self.outdir, layer, self.nPools, self.nPools), features) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir', default='/scratch/groups/jlg/texture_stimuli/color')
    parser.add_argument('-o', '--outdir', default='/scratch/groups/jlg/texture_stimuli/color/deepnet_features2')
    parser.add_argument('-f', '--featuretype', default='activations')
    parser.add_argument('-n', '--nPools', type=int, default=1)
    args = parser.parse_args()
 
    VGG19_FeatureExtractor(feature_type=args.featuretype, nPools=args.nPools, inputdir=args.inputdir, outdir = args.outdir)

