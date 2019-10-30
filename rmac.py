from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
#from vgg16 import VGG16
from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map
import scipy.io
import numpy as np
import utils
import os
from cv2 import cv2
from PIL import Image
import pickle
from keras.applications.vgg19 import VGG19

def addition(x):
    sum = K.sum(x, axis=1)
    return sum

def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, 512, axis=-1)
    out = x * w
    return out

def rmac(input_shape1, num_rois):
    
    # Load VGG16
    #vgg16_model = VGG16(utils.DATA_DIR + utils.WEIGHTS_FILE, input_shape1)
    vgg16_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape1 , pooling=None)
    
    # Regions as input
    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    # ROI pooling
    x = RoiPooling([1], num_rois)([vgg16_model.layers[-1].output, in_roi])

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

    # PCA
    x = TimeDistributed(Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(512,), name='rmac')(x)

    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)

    # Define model
    model = Model([vgg16_model.input, in_roi], rmac_norm)

    # Load PCA weights
    mat = scipy.io.loadmat(utils.DATA_DIR + utils.PCA_FILE)
    b = np.squeeze(mat['bias'], axis=1)
    w = np.transpose(mat['weights'])
    model.layers[-4].set_weights([w, b])

    return model

def get_rmac(imag):
    
    img = imag
    scale = utils.IMG_SIZE / max(img.size)
    new_size =(int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    print('Original size: %s, Resized image: %s' %(str(img.size), str(new_size)))
    img = img.resize(new_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x=x/255.
    #x = utils.preprocess_image(x)
    
    # Load RMAC model
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[2], x.shape[1])
    regions = rmac_regions(Wmap, Hmap, 3)
    print('Loading RMAC model...')
    model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))

    # Compute RMAC vector
    print('Extracting RMAC from image...')
    RMAC = model.predict([x, np.expand_dims(regions, axis=0)])
    return RMAC

if __name__ == "__main__":
    
    photo_number=0
    learned_codes=[]
    DATADIR = "D:/paris_1/paris/"
    CATEGORIES = ["defense","eiffel","general","invalides","louvre","moulinrouge",
                                    "museedorsay","notredame","pantheon","pompidou","sacrecoeur","triomphe"]
    for category in CATEGORIES: 
        
        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  
        
        for img in (os.listdir(path))[50:100]:
            
            imag = image.load_img(os.path.join(path,img))
            #img = cv2.imread(os.path.join(path,img)) 
            try:
                RMAC = get_rmac(imag)
                learned_codes.append(RMAC)
                print('RMAC size: %s' % RMAC.shape[1])
                print('Done!')
                photo_number=photo_number+1
                print(photo_number)
            except:
                photo_number=photo_number+1
                print(photo_number)
                print(os.path.join(path,img))
                continue 
                
    learned_codes=np.array(learned_codes)
    pickle_out = open("learned_codes.pickle","wb")
    pickle.dump(learned_codes, pickle_out)
    pickle_out.close()