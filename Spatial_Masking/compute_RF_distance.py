import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
import os

from RFModel import RFModel
from model import VGG19

#################3
# Helper Functions
# ########
def preprocess_im(path):
    '''
    Function for preprocessing images by subtracting VGG Mean
    '''
    MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    image = imread(path)

    if image.shape[1]!=256 or image.shape[0]!=256:
        image = resize(image, (256,256))

    # Resize the image for convnet input, add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    if len(image.shape)<4:
        image = np.stack((image,image,image),axis=3)

    # If there is a Alpha channel, just scrap it
    if image.shape[3] == 4:
        image = image[:,:,:,:3]

    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image

def load_guides(guide_dir, imSize=256):
    '''
    Given a directory containing luminance mask images, this function loads each image and returns
    an array of guides as well as the filenames in the corresponding order they were loaded.
     - guide_dir : string path to directory containing mask files (ALL FILES IN THIS DIRECTORY must be masks)
     - imSize : size of original images -- masks will be rescaled to fit this.
    '''
    nGuides = len(os.listdir(guide_dir))
    guides = np.zeros((256, 256, nGuides))
    guideNames = []
    for i, imName in enumerate(os.listdir(guide_dir)[:nGuides]):
        guideI = resize(imread('{}/{}'.format(guide_dir, imName)), (256,256))
        guides[:,:,i] = guideI
        guideNames.append(imName)
    return guides, guideNames


def load_vgg():
    # Load VGG19 Weights and build TF model
    weights_file = 'vgg19_normalized.pkl'
    with open(weights_file, 'rb') as f:
        vgg_weights = pickle.load(f)['param values']
    vgg19 = VGG19(vgg_weights)
    vgg19.build_model()
    return vgg19
 

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

poolsize = '3x3'
scaling = 0.70
layers = ['conv1_1' ,'pool1', 'pool2', 'pool3', 'pool4']
inputdir = '/scratch/groups/jlg/texture_stimuli/color/originals'
texture_dir = '/scratch/groups/jlg/texture_stimuli/color/textures'
guide_dir = '/home/users/akshayj/TextureSynthesis/tensorflow_synthesis/Receptive_Fields/MetaWindows_clean_s{}'.format(scaling)
rf_feature_dir = '/scratch/groups/jlg/RF_Features'

# Load VGG-19 weights and guide images.
vgg19 = load_vgg()
guides, guidenames = load_guides(guide_dir)

# Initialize the Receptive Field model.
rf_model = RFModel(vgg19, guides, layers)

for image in ['elephant', 'jetplane', 'dirt']:
    # Load each image and preprocess
    image_path = '{}/{}.jpg'.format(inputdir, image)
    original_image = preprocess_im(image_path)

    # Extract the gramian for each image
    gramian = rf_model.get_gramian(original_image, layers)
    gramian['guidenames'] = guidenames
    np.save('{}/Scaling{}-{}_orig.npy'.format(rf_feature_dir, scaling, image), gramian)

    for poolsize in ['1x1', '2x2', '4x4']:
        for layer in ['pool1', 'pool2', 'pool4']:
            tex_path = '{}/{}_{}_{}_smp1.png'.format(texture_dir, poolsize, layer, image)
            tex_image = preprocess_im(tex_path)

            gram = rf_model.get_gramian(tex_image, layers)
            gram['guidenames'] = guidenames

            np.save('{}/Scaling{}-{}_{}_{}_smp1.npy'.format(rf_feature_dir, scaling, poolsize, layer, image), gram)

