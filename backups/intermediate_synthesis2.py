import numpy as np
import tensorflow as tf
import sys

from ImageUtils import *
from model import *
from skimage.io import imread, imsave

SAVE_STEP = 1000

class TextureSynthesis:
    def __init__(self, sess, model, actual_image1, actual_image2, proportion1, proportion2, layer_constraints, layer_name, image_name, saveDir, iterations, nSplits):
        # 'layer_constraints' is dictionary with key = VGG layer and value = weight (w_l)
        # 'sess' is tensorflow session
        self.layer_name = layer_name # Of the form: conv#
        self.image_name = image_name # Of the form: imageName

        self.sess = sess
        self.sess.run(tf.initialize_all_variables())

        self.model = model # Model instance
        assert self.model.model_initialized(), "Model not created yet."
        self.model_layers = self.model.get_model()

        # Layer weights for the loss function
        self.layer_weights = layer_constraints

        self.actual_image1 = actual_image1 # 256x256x3
        self.actual_image2 = actual_image2
        self.proportion1 = proportion1
        self.proportion2 = proportion2

        self.init_image = self._gen_noise_image()

        # Number of splits
        self.nSpl = nSplits

        # Directory to save outputs in
        self.saveDir = saveDir

        # Number of iterations to run for
        self.iterations = iterations

        # Image size and network output size
        self.imsize = self.actual_image1.shape[1]
        self.net_size = self.get_net_size()

        # Get subset boundaries
        self.subset_boundaries = self.get_subset_boundaries()

        # precompute layer subset weights
        print 'Precomputing layer subset weights'
        self.layer_subset_weights = self.precompute_layer_subset_weights()

        # Get constraints
        self.constraints = self._get_constraints() # {layer_name: activations}


    def get_texture_loss(self):
        total_loss = 0.0
        for layer in self.layer_weights.keys():
            print layer
            layer_activations = self.model_layers[layer]
            layer_activations_shape = layer_activations.get_shape().as_list()
            assert len(layer_activations_shape) == 4 # (1, H, W, outputs)
            assert layer_activations_shape[0] == 1, "Only supports 1 image at a time."
            num_filters = layer_activations_shape[3] # N
            num_spatial_locations = layer_activations_shape[1] * layer_activations_shape[2] # M
            layer_gram_matrix = self._compute_weighted_gram_matrix(layer, layer_activations, num_filters, num_spatial_locations)
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
            gram_matrix1= self._compute_weighted_gram_matrix_np(layer, layer_activations, num_filters, num_spatial_locations)

            self.sess.run(self.model_layers['input'].assign(self.actual_image2))
            layer_activations = self.sess.run(self.model_layers[layer])
            num_filters = layer_activations.shape[3] # N
            num_spatial_locations = layer_activations.shape[1] * layer_activations.shape[2] # M
            print layer_activations.shape
            gram_matrix2 = self._compute_weighted_gram_matrix_np(layer, layer_activations, num_filters, num_spatial_locations)
            
            constraints[layer] = self.proportion1*gram_matrix1 + self.proportion2*gram_matrix2;

        return constraints

    def _compute_gram_matrix_np(self, F, N, M):
        F = F.reshape(M, N)
        return np.dot(F.T, F)

    def _compute_gram_matrix(self, layer, F, N, M):
        # F: (1, height, width, num_filters), layer activations
        # N: num_filters
        # M: number of spatial locations in filter (filter size ** 2)
        F = tf.reshape(F, (M, N)) # Vectorize each filter so F is now of shape: (height*width, num_filters)
        return tf.matmul(tf.transpose(F), F)  

    def get_subset_boundaries(self):
        '''
        Using self.nSpl (number of splits), computes the subset boundaries
        - returns a list of lists each containing the boundaries of the i'th subset 
        - subset is aka gramRF aka pooling region
        '''
        sub_sz = int(self.imsize / self.nSpl)
        subset_boundaries = []

        for hi in range(0, self.imsize, sub_sz):
            end_h = np.minimum(hi + sub_sz, self.imsize)
            for wi in range(0, self.imsize, sub_sz):
                end_w = np.minimum(wi + sub_sz, self.imsize)
                sub_bound = [[hi, wi], [end_h, end_w]]

                subset_boundaries.append(sub_bound)

        return subset_boundaries
    
    def _compute_weighted_gram_matrix(self, layer, F, N, M):
        '''
        Computes gram matrix
        '''
        F2 = tf.to_float(tf.constant(np.zeros(N*N), shape=(N, N, 1)));
        F = tf.reshape(F, (M, N))
        weight_mtx = self.layer_subset_weights[layer]

        for si in range(len(self.subset_boundaries)):
            subset_weights = tf.to_float(tf.reshape(weight_mtx[:,:,si], (M,1)))
            weighted_F = tf.multiply(F, subset_weights)

            dp = tf.matmul(tf.transpose(weighted_F), weighted_F)
            dp = tf.reshape(dp, (N,N,1))
            F2 = tf.concat(values=[F2, dp], axis=2)

        return F2[:,:,1:]

    def _compute_weighted_gram_matrix_np(self, layer, F, N, M):
        '''
        Computes gram matrix weighted by the proportion that each unit's RF overlaps with each subset.
        '''
        F2 = np.zeros((N,N,1))
        F = np.reshape(F, (M,N))
        weight_mtx = self.layer_subset_weights[layer]

        for si in range(len(self.subset_boundaries)):
            subset_weights = np.reshape(weight_mtx[:,:,si], (M,1))
            weighted_F = np.multiply(F, subset_weights)

            print weighted_F.shape

            dp = np.matmul(weighted_F.T, weighted_F).reshape((N,N,1))
            F2 = np.concatenate((F2,dp), axis=2)

        return F2[:,:,1:]

    def precompute_layer_subset_weights(self, imsize=256):
        '''
          Precompute layer subset weights
              Returns a dictionary with keys = layer names, and values = matrix of 
              (out_size x out_size x nSubsets) where each element corresponds to the weight.
        '''
        layer_names = ['conv1_1', 'conv1_2','pool1','conv2_1', 'conv2_2', 'pool2', 'conv3_1', \
                       'conv3_2','pool3','conv4_1','conv4_2','conv4_3','conv4_4','pool4',\
                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5']
        lsub_weights = {}
        for layI in range(len(layer_names)):
            lname = layer_names[layI]
            out_size = int(self.net_size['out_size'][layI])
            
            nSubsets = len(self.subset_boundaries)
            layer_weight = np.zeros((out_size, out_size, nSubsets))

            for si in range(nSubsets):
                isub = self.subset_boundaries[si]
                for xi in range(out_size):
                    for yi in range(out_size):
                        pos = [xi,yi]
                        rf_size, center, [tl,br] = self.get_rf_coords(lname, pos)
                        layer_weight[xi,yi,si] = calc_proportion_overlap([tl,br], isub, self.imsize)
            lsub_weights[lname] = layer_weight
        return lsub_weights

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
        for i in range(self.iterations):
            self.sess.run(train_step)
            if i % 50 == 0:
                print "Iteration: " + str(i) + "; Loss: " + str(self.sess.run(content_loss))
            if i % SAVE_STEP == 0:
                print "Saving image..."
                curr_img = self.sess.run(self.model_layers["input"])
                filename = self.saveDir + "/iters/%gx%g_%s_%s_%s_step_%d" % (self.nSpl, self.nSpl, self.layer_name, self.image_name, props, i)
                print "Saving image as " + filename + '...'
                save_image(filename, curr_img)
            sys.stdout.flush()
        filename = self.saveDir + "/%gx%g_%s_%s_%s" % (self.nSpl, self.nSpl, self.layer_name, self.image_name, props)
        save_image(filename, curr_img)
        a = np.load(filename + '.npy')
        imsave(filename + '.png', a)

    
    def get_net_size(self):
        # Net = vgg architecture [filter size, stride, padding]
        net = [[3,1,1],[3,1,1], [2,2,0], [3,1,1], [3,1,1], [2,2,0], [3,1,1], [3,1,1],[2,2,0],\
               [3,1,1], [3,1,1], [3,1,1], [3,1,1],[2,2,0], [3,1,1], [3,1,1], [3,1,1], [3,1,1], [2,2,0]]

        out_size = np.zeros(len(net))
        rf_size = np.zeros(len(net))
        tot_stride = np.zeros(len(net))
        start1 = np.zeros(len(net), dtype=np.float64)
        
        insize = self.imsize
        totstride = 1
        startPos = 0.5
        rf_sz = net[0][0]
        for layer in range(len(net)):
            filt_sz, stride, pad = net[layer]
            
            # Calculate outsize as a function of insize
            out_sz = np.floor((insize - filt_sz + 2*pad) / stride) + 1

            actualP = (out_sz-1)*stride - insize + filt_sz
            pL = np.floor(actualP/2)

            # Calculate RF size as a function of previous layer RF size
            if layer > 0:
                rf_sz = rf_sz + (filt_sz-1)*totstride
                
            # Start position
            startPos = startPos + ((filt_sz-1)/2.0 - pL)*totstride
            
            # Distance between the center of adjacent features 
            totstride = totstride * stride
            
            out_size[layer], rf_size[layer], tot_stride[layer] = out_sz, rf_sz, totstride
            start1[layer] = startPos
            
            insize = out_sz
            
        net_size = {'out_size': out_size, 'rf_size': rf_size, 'tot_stride': tot_stride, 'center00': start1}
        return net_size
    
    def get_rf_coords(self, layerName, pos, imsize=256):
        layer_names = ['conv1_1', 'conv1_2','pool1','conv2_1', 'conv2_2', 'pool2', 'conv3_1', \
                  'conv3_2','pool3','conv4_1','conv4_2','conv4_3','conv4_4','pool4',\
                  'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5']

        lay_idx = layer_names.index(layerName)
        
        rf_size = self.net_size['rf_size'][lay_idx]
        start = self.net_size['center00'][lay_idx]
        jump = self.net_size['tot_stride'][lay_idx]

        center = [start+pos[0]*jump, start + pos[1]*jump]
        top_left = [center[0] - rf_size/2.0, center[1] - rf_size/2.0]
        bottom_right = [center[0] + rf_size/2.0, center[1] + rf_size/2.0]

        return rf_size, center, [top_left, bottom_right]

def calc_proportion_overlap(rf, subset, imsize): 
    '''
    Calculates what percentage of the receptive field is contained within the subset
       - takes as arguments a neuron's RF, a gram RF (aka subset), and the image size.
    '''
    tl_subset, br_subset = subset[0], subset[1] # rect1
    tl_rf, br_rf = rf[0], rf[1] # rect2
    
    # Total receptive field area
    rf_area = (br_rf[0] - tl_rf[0])*(br_rf[1] - tl_rf[1])
    
    ## OLD METHOD
    #x_overlap = np.maximum(0, np.minimum(br_subset[0], br_rf[0]) - np.maximum(tl_subset[0], tl_rf[0]));
    #y_overlap = np.maximum(0, np.minimum(br_subset[1], br_rf[1]) - np.maximum(tl_subset[1], tl_rf[1]));
    #overlapArea = 1.0*x_overlap * y_overlap / rf_area;

    # New method: trapezoidal receptive fields
    rf_mtx = np.zeros((imsize, imsize))
    sub_mtx = np.zeros((imsize, imsize))
    rf_mtx[int(np.maximum(0,tl_rf[0])):int(np.minimum(imsize,br_rf[0])), int(np.maximum(0,tl_rf[1])):int(np.minimum(imsize,br_rf[1]))] = 1
    xlim = [int(tl_subset[0]), int(br_subset[0])]
    ylim = [int(tl_subset[1]), int(br_subset[1])]
    sub_mtx = calc_subset_shape(imsize, xlim, ylim)

    ## OLD METHOD 2: Square receptive fields
    #sub_mtx[xlim[0]:xlim[1], ylim[0]:ylim[1]] = 1

    overlap2 = np.sum(np.multiply(rf_mtx, sub_mtx))*1.0 / rf_area

    return overlap2

def calc_subset_shape(imsize, xlim, ylim):
    # Make subsets shaped as trapezoids (flat top, angular borders)
    # mrgn sets the size of the borders
    mrgn = 10
    rpmt = np.matlib.repmat
    imin = lambda x,y: int(np.minimum(x,y))

    sub_mtx = np.zeros((imsize,imsize))
    
    # Top of trapezoid
    sub_mtx[xlim[0]: xlim[1], ylim[0]: ylim[1]] = 1

    # left side
    if xlim[0] != 0:
        brdr = np.linspace(0,1, imin(xlim[0]+mrgn, imsize) - (xlim[0]-mrgn))
        sub_mtx[xlim[0]-mrgn:xlim[0]+mrgn, ylim[0]:ylim[1]] = rpmt(brdr, ylim[1]-ylim[0], 1).T
    # top side
    if ylim[0] != 0:
        brdr = np.linspace(0,1, imin(ylim[0]+mrgn, imsize) - (ylim[0]-mrgn))
        sub_mtx[xlim[0]:xlim[1], ylim[0]-mrgn:ylim[0]+mrgn] = rpmt(brdr, xlim[1]-xlim[0], 1)
    # right side
    if xlim[1] != imsize:
        brdr2 = np.linspace(1,0, imin(xlim[1]+mrgn, imsize) - (xlim[1]-mrgn))
        sub_mtx[xlim[1]-mrgn:xlim[1]+mrgn, ylim[0]:ylim[1]] = rpmt(brdr2, ylim[1]-ylim[0],1).T
    # bottom side
    if ylim[1] != imsize:
        brdr2 = np.linspace(1,0, imin(ylim[1]+mrgn, imsize) - (ylim[1]-mrgn))
        sub_mtx[xlim[0]:xlim[1], ylim[1]-mrgn:ylim[1]+mrgn] = rpmt(brdr2, xlim[1]-xlim[0],1)

    return sub_mtx

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

#===============================
if __name__ == "__main__":
    args = sys.argv
    img1 = "%s.npy" %(args[1])
    img2 = "%s.npy" %(args[2])
    proportion1 = float(args[3])
    proportion2 = float(args[4])

    sess = tf.Session()
    layer_weights = {'conv1_1': 1e9, 'pool1': 1e9, 'pool2': 1e9, 'pool3': 1e9, 'pool4': 1e9}
    layer_name = 'pool2'
    vgg_weights = VGGWeights('vgg_synthesis/vgg19_normalized.pkl')
    my_model = Model(vgg_weights)
    my_model.build_model()
    image_name = args[1] + '_' + args[2]

    image1 = np.load('/scratch/groups/jlg/grant/orig_all/%s' %(img1))
    image2 = np.load('/scratch/groups/jlg/grant/orig_all/%s' %(img2))

    saveDir = '/scratch/groups/jlg/intermediate'
    iterations = 5001
    nSplits = 4
    ts = TextureSynthesis(sess, my_model, image1, image2, proportion1, proportion2, layer_weights, layer_name, image_name, saveDir, iterations, nSplits)
    #np.save('/home/users/akshayj/TextureSynthesis/tmp/gram_%s.npy' %(image_name), ts.constraints)

    print 'Success in initializing.. Training....'
    ts.train()

