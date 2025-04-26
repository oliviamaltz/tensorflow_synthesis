import numpy as np
import tensorflow as tf
import sys, os
import time

from collections import OrderedDict
from ImageUtils import *
from model import *


class SpatialTextureSynthesis:
    def __init__(self, model, original_image, guides, style_loss_layer_weights, saveParams, iterations=5000, diag=0):
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

        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        # Layer weights for the loss function
        self.style_loss_layer_weights = style_loss_layer_weights
        self.original_image = original_image # 256x256x3

        # Directory to save outputs in
        if 'saveDir' in saveParams:
            self.saveDir = saveParams['saveDir']
        else:
            self.saveDir = '.'

        if 'saveName' in saveParams:
            self.saveName = saveParams['saveName']
        else:
            self.saveName = 'unnamedtexture'

        # Number of iterations to run for
        self.iterations = iterations

        # Diagonal or Gramian?
        self.diag = diag

        # Image size and network output size
        self.imsize = self.original_image.shape[1]

        # Guides
        if guides is None or guides == 'all':
            self.guides = np.squeeze(np.ones(original_image.shape))
        else:
            self.guides = guides
        print('Shape of guides: {}'.format(self.guides.shape))
        self.fm_guides = self.get_fm_guides(layers=self.style_loss_layer_weights, mode='all')
        for layer in self.style_loss_layer_weights: #normalise fm guides
            denom = np.expand_dims(np.expand_dims(np.sqrt(np.diag(self.gram_matrix(np.moveaxis(self.fm_guides[layer], 0, -1)))), -1), -1)
            self.fm_guides[layer] = self.fm_guides[layer]/denom
            #self.fm_guides[layer] = self.fm_guides[layer]/np.sqrt(np.diag(self.gram_matrix(self.fm_guides[layer])))

        # # Get constraints
        self.gramian = self.get_gramian() # {layer_name: activations}

    def get_fm_guides(self, layers, mode='inside', th=0.5, batch_size=2):
        fm_guides = OrderedDict()
        n_guides = self.guides.shape[2]
        for m in range(n_guides):
            guide = self.guides[:,:,m]
            guide[guide<th] = 0
            guide[guide>=th] = 1

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

    def get_texture_loss(self):
        total_loss = 0.0
        for layer in self.style_loss_layer_weights.keys():
            print(layer)
            layer_activations = self.model_layers[layer]
            layer_activations_shape = layer_activations.get_shape().as_list()
            num_filters = layer_activations_shape[3] # N
            num_spatial_locations = layer_activations_shape[1] * layer_activations_shape[2] # M
            layer_gram_matrix = self.gram_matrix_guided_tf(layer_activations, self.fm_guides[layer])
            desired_gram_matrix = self.gramian[layer]

            total_loss += self.style_loss_layer_weights[layer] * (1.0 / (4*num_filters**2) * (num_spatial_locations**2)) * tf.reduce_sum(tf.pow(desired_gram_matrix - layer_gram_matrix, 2))
        return total_loss

    def get_gramian(self, image=None):
        if image is None:
            image = self.original_image
        self.sess.run(tf.initialize_all_variables())
        gramian = dict()
        for layer in self.style_loss_layer_weights:
            self.sess.run(self.model_layers['input'].assign(image))
            layer_activations = self.sess.run(self.model_layers[layer])
            gramian[layer] = self.gram_matrix_guided(layer_activations, self.fm_guides[layer])
        return gramian

    def get_activations(self, image=None, layers=None):
        if layers is None:
            layers = self.style_loss_layer_weights
        if image is None:
            image = self.original_image
        
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

    def gram_matrix_guided_tf(self, activations, guides):
        '''
        guides is array of dimensions (n_ch,h,w) that defines n_ch guidance channels
        guides should be normalised as: guides = guides / np.sqrt(np.diag(gram_matrix(guides)))[:,None,None]
        activations are of dimensions (n_fm,h,w), the n_fm feature maps of a CNN layer
        Output are n_ch gram matrices, that were computed with the feature maps weighted by the guidance channel
        '''
        print('(gram_matrix_guided_tf) Activations Shape: {}, Guides Shape: {}'.format(activations.shape, guides.shape))
        n_pos = activations.shape[1]*activations.shape[2]
        n_fm = activations.shape[-1]    # number of feature maps
        n_ch = guides.shape[0]          # number of guidance channels
        if self.diag==1:
            G = tf.zeros((n_fm,1))
        else:
            G = tf.zeros((n_fm,n_fm,1))
        for c in range(n_ch):
            F = tf.multiply(tf.squeeze(activations), tf.cast(tf.expand_dims(guides[c,:,:],-1), tf.float32))
            F = tf.reshape(tf.transpose(F, perm=[2,0,1]), (n_fm,n_pos))

            F2 = tf.matmul(F, tf.transpose(F)) / tf.cast(n_pos, tf.float32)
            if self.diag==1:
                diag = tf.expand_dims(tf.diag_part(F2), -1)
                print(diag.shape)
                G = tf.concat(values=[G,diag], axis=1)
            else:
                F2 = tf.expand_dims(F2, -1)
                G = tf.concat(values=[G,F2], axis=2)
        print(G.shape)
        if self.diag==1:
            return G[:,1:]
        else:
            return G[:,:,1:]

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
        if self.diag == 1:
            G = np.zeros((n_fm, n_ch))
        else:
            G = np.zeros((n_fm,n_fm,n_ch))
        for c in range(n_ch):
            # First do an element-wise multiplication on 
            F = np.moveaxis(np.squeeze(activations) * np.expand_dims(guides[c,:,:],-1), -1, 0).reshape(n_fm,-1)
            gram = F.dot(F.T) / n_pos
            if self.diag == 1:
                G[:,c] = np.diag(gram)
            else:
                G[:,:,c] = gram

        return G

    def _gen_noise_image(self):
        input_size = self.model_layers["input"].get_shape().as_list()
        return np.random.randn(input_size[0], input_size[1], input_size[2], input_size[3])

    def train(self, sampleIdx=1, SAVE_STEP=1000):
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.model_layers["input"].assign(self.original_image))

        content_loss = self.get_texture_loss()
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(content_loss)
        self.init_image = self._gen_noise_image()

        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.model_layers["input"].assign(self.init_image))
        for i in range(self.iterations):
            self.sess.run(train_step)
            if i % 10 == 0:
                print("Iteration: " + str(i) + "; Loss: " + str(self.sess.run(content_loss)))
            if i % SAVE_STEP == 0:
                print("Saving image...")
                curr_img = self.sess.run(self.model_layers["input"])
                filename = "{}/{}_smp{}_step_{}".format(self.saveDir, self.saveName, sampleIdx, i)
                save_image(filename, curr_img)
            sys.stdout.flush()
    
def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

#===============================
if __name__ == "__main__":
    args = sys.argv
    
