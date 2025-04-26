import numpy as np
import tensorflow as tf
import sys, os
import time

from collections import OrderedDict
from ImageUtils import *
from model import VGG19

class RFModel:
    def __init__(self, model, guides, layers):
        '''
        Initializes a Spatial Texture Synthesis object

        - Required Arguments:
            - 'model': this is a Tensorflow model which should be defined like model.py
            - 'original_image': a 256x256x3 image 
         'style_loss_layer_weights' is dictionary with key = VGG layer and value = weight (w_l)
        'sess' is tensorflow session

        '''

        # Get the model and layers.
        self.model = model # Model instance
        assert self.model.model_initialized(), "Model not created yet."
        self.model_layers = self.model.get_model()
        self.layers = layers

        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        # Guides
        if guides is None or guides == 'all':
            self.guides = np.squeeze(np.ones(original_image.shape))
        else:
            self.guides = guides
        print('Shape of guides: {}'.format(self.guides.shape))
        self.fm_guides = self.get_fm_guides(layers=self.layers, mode='all')
        for layer in self.layers: #normalise fm guides
            denom = np.expand_dims(np.expand_dims(np.sqrt(np.diag(self.gram_matrix(np.moveaxis(self.fm_guides[layer], 0, -1)))), -1), -1)
            #self.fm_guides[layer] = self.fm_guides[layer]/denom

    def get_fm_guides(self, layers, mode='all', th=0.5, batch_size=2):
        fm_guides = OrderedDict()
        n_guides = self.guides.shape[2]
        for m in range(n_guides):
            guide = self.guides[:,:,m]
            #guide[guide<th] = 0
            #guide[guide>=th] = 1

            if mode=='all':
                probe_image = np.zeros((batch_size,) + guide.shape + (3,))
                probe_image[:,guide.astype(bool),:] += 1e2 * np.random.randn(*probe_image[:,guide.astype(bool),:].shape)
                feature_maps = self.get_activations(image=probe_image, layers=layers)
                for layer in layers:
                    if m==0:
                        fm_guides[layer] = []
                    fm_guides[layer].append((np.nanmean(np.nanvar(feature_maps[layer], axis=0), axis=-1)!=0).astype(float))

            elif mode=='inside':
                inv_guide = guide.copy()-1
                inv_guide *= -1
                probe_image_out = np.zeros((batch_size,) + inv_guide.shape + (3,))
                probe_image_out[:,inv_guide.astype(bool), :] += 1e2 * np.random.randn(*probe_image_out[:,inv_guide.astype(bool),:].shape)
                feature_maps_out = self.get_activations(image=probe_image_out, layers=layers)

                for layer in layers:
                    if m==0:
                        fm_guides[layer] = []
                    fm_guides[layer].append((np.nanmean(np.nanvar(feature_maps_out[layer], axis=0), axis=-1)==0).astype(float))
        for layer in layers:
            fm_guides[layer] = np.stack(fm_guides[layer])
        return fm_guides
        
            
    def get_gramian(self, image, layers):
        self.sess.run(tf.initialize_all_variables())
        gramian = OrderedDict()
        for layer in layers:
            self.sess.run(self.model_layers['input'].assign(image))
            layer_activations = self.sess.run(self.model_layers[layer])
            gramian[layer] = self.gram_matrix_guided(layer_activations, self.fm_guides[layer])
        return gramian

    def get_activations(self, image, layers):
        
        self.sess.run(tf.initialize_all_variables())
        activations = OrderedDict()
        for layer in layers:
            activations[layer] = []
            for i in range(image.shape[0]):
                imgI = np.expand_dims(image[i,:,:,:],0)
                self.sess.run(self.model_layers['input'].assign(imgI))
                activations[layer].append(self.sess.run(self.model_layers[layer]))
            activations[layer] = np.concatenate(activations[layer], axis=0)
        return activations

    def gram_matrix(self, activations):
        n_fm = activations.shape[-1]
        F = activations.reshape(n_fm,-1)
        G = F.dot(F.T) / F[0,:].size
        return G

    def gram_matrix_guided(self, activations, guides):
        '''
        guides is array of dimensions (n_ch,h,w) that defines n_ch guidance channels
        guides should be normalised as: guides = guides / np.sqrt(np.diag(gram_matrix(guides)))[:,None,None]
        activations are of dimensions (n_fm,h,w), the n_fm feature maps of a CNN layer
        Output are n_ch gram matrices, that were computed with the feature maps weighted by the guidance channel
        '''
        print('gram_matrix_guided_np: Activations Shape = {}; Guides Shape = {}'.format(activations.shape, guides.shape))
        n_pos = activations.shape[1]*activations.shape[2]
        n_fm = activations.shape[-1]    # number of feature maps
        n_ch = guides.shape[0]          # number of guidance channels
        G = np.zeros((n_fm,n_fm,n_ch))
        for c in range(n_ch):
            # First do an element-wise multiplication on 
            F = np.moveaxis(np.squeeze(activations) * np.expand_dims(guides[c,:,:],-1), -1, 0).reshape(n_fm,-1)
            G[:,:,c] = F.dot(F.T) / n_pos
        return G

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

#===============================
if __name__ == "__main__":
    args = sys.argv

