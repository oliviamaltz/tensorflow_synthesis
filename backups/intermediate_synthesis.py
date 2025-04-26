import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

from ImageUtils import *
from model import *
from skimage.io import imread, imsave

SAVE_STEP = 1000
ITERATIONS = 5001

class TextureSynthesis:
    def __init__(self, sess, model, actual_image1, actual_image2, proportion1, proportion2, layer_constraints, layer_name, image_name, saveDir):
        # 'layer_constraints' is dictionary with key = VGG layer and value = weight (w_l)
        # 'sess' is tensorflow session
        self.layer_name = layer_name # Of the form: conv#
        self.image_name = image_name # Of the form: imageName

        self.sess = sess
        self.sess.run(tf.initialize_all_variables())

        self.saveDir = saveDir
        self.model = model # Model instance
        assert self.model.model_initialized(), "Model not created yet."
        self.model_layers = self.model.get_model()

        # Layer weights for the loss function
        self.layer_weights = layer_constraints

        self.actual_image1 = actual_image1 # 256x256x3
        self.actual_image2 = actual_image2 # 256x256x3
        self.proportion1 = proportion1
        self.proportion2 = proportion2

        self.init_image = self._gen_noise_image()
        self.constraints = self._get_constraints() # {layer_name: activations}


    def get_texture_loss(self):
        total_loss = 0.0
        for layer in self.layer_weights.keys():
            layer_activations = self.model_layers[layer]
            layer_activations_shape = layer_activations.get_shape().as_list()
            assert len(layer_activations_shape) == 4 # (1, H, W, outputs)
            assert layer_activations_shape[0] == 1, "Only supports 1 image at a time."
            num_filters = layer_activations_shape[3] # N
            num_spatial_locations = layer_activations_shape[1] * layer_activations_shape[2] # M
            layer_gram_matrix = self._compute_gram_matrix(layer_activations, num_filters, num_spatial_locations)
            desired_gram_matrix = self.constraints[layer]

            total_loss += self.layer_weights[layer] * (1.0 / (4 * (num_filters**2) * (num_spatial_locations**2))) \
                          * tf.reduce_sum(tf.pow(desired_gram_matrix - layer_gram_matrix, 2))
        return total_loss

    def _get_constraints(self):
        self.sess.run(tf.initialize_all_variables())
        constraints = dict()
        for layer in self.layer_weights:
            self.sess.run(self.model_layers['input'].assign(self.actual_image1))
            layer_activations = self.sess.run(self.model_layers[layer])
            num_filters = layer_activations.shape[3] # N
            num_spatial_locations = layer_activations.shape[1] * layer_activations.shape[2] # M
            gram_matrix1 = self._compute_gram_matrix_np(layer_activations, num_filters, num_spatial_locations)

            self.sess.run(self.model_layers['input'].assign(self.actual_image2))
            layer_activations = self.sess.run(self.model_layers[layer])
            num_filters = layer_activations.shape[3]
            num_spatial_locations = layer_activations.shape[1]*layer_activations.shape[2]
            gram_matrix2 = self._compute_gram_matrix_np(layer_activations, num_filters, num_spatial_locations)

            constraints[layer] = self.proportion1*gram_matrix1 + self.proportion2*gram_matrix2;

        return constraints

    def _compute_gram_matrix_np(self, F, N, M):
        F = F.reshape(M, N)
        return np.dot(F.T, F)

    def _compute_gram_matrix(self, F, N, M):
        # F: (1, height, width, num_filters), layer activations
        # N: num_filters
        # M: number of spatial locations in filter (filter size ** 2)
        F = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(F), F)

    def _gen_noise_image(self):
        input_size = self.model_layers["input"].get_shape().as_list()
        return np.random.randn(input_size[0], input_size[1], input_size[2], input_size[3])

    def train(self):
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.model_layers["input"].assign(self.actual_image1))

        content_loss = self.get_texture_loss()
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(content_loss)

        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.model_layers["input"].assign(self.init_image))
        props = "%i-%i" % (100*self.proportion1, 100*self.proportion2)
        for i in range(ITERATIONS):
            self.sess.run(train_step)
            if i % 50 == 0:
                print "Iteration: " + str(i) + "; Loss: " + str(self.sess.run(content_loss))
            if i % SAVE_STEP == 0:
                curr_img = self.sess.run(self.model_layers["input"])
                filename = self.saveDir + "/iters/%s_%s_%s_step_%d" % (self.layer_name, self.image_name, props, i)
                print "Saving image as " + filename + "..."
                save_image(filename, curr_img)
            sys.stdout.flush()
        filename = self.saveDir + '/%s_%s_%s' % (self.layer_name, self.image_name, props)
        save_image(filename, curr_img) # save as npy
        a = np.load(filename + '.npy')
        imsave(filename + '.png', a) # and save as png

#===============================
def gram_matrix(x, area, depth):
    x1 = tf.reshape(x,(area,depth))
    g = tf.matmul(tf.transpose(x1), x1)
    return g

def gram_matrix_val(x, area, depth):
    x1 = x.reshape(area,depth)
    g = np.dot(x1.T, x1)
    return g

def build_style_loss(a, x):
    M = a.shape[1]*a.shape[2]
    N = a.shape[3]
    A = gram_matrix_val(a, M, N )
    G = gram_matrix(x, M, N )
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
    return loss
#===============================


if __name__ == "__main__":
    args = sys.argv
    img1 = "%s.npy" % (args[1])
    img2 = "%s.npy" % (args[2])
    proportion1 = float(args[3])
    proportion2 = float(args[4])

    vgg_weights = VGGWeights('vgg_synthesis/vgg19_normalized.pkl')
    my_model = Model(vgg_weights)
    my_model.build_model()

    image1 = np.load("/scratch/groups/jlg/grant/orig_all/%s" % (img1) )
    image2 = np.load("/scratch/groups/jlg/grant/orig_all/%s" % (img2) )

    sess = tf.Session()
    layer_weights = {'conv1_1': 1e9, 'pool1':1e9, 'pool2': 1e9, 'pool3':1e9, 'pool4':1e9}
    layer_name = "pool2"
    image_name = args[1] + '_' + args[2]

    print "Synthesizing textures for image " + image_name + " with proportions " + str(proportion1) + " and " + str(proportion2)
    saveDir = '/scratch/groups/jlg/intermediate'
    text_synth = TextureSynthesis(sess, my_model, image1, image2, proportion1, proportion2, layer_weights, layer_name, image_name, saveDir)

    print "Success in initializing."
    print "Training..."

    text_synth.train()

