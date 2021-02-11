#feater extraction code using Rmac pooling
#INPUT:each image per each class in Paris 6K 
#OUTPUT: respective (extracted featuers + image name) for each input image 
import numpy as np
from keras.engine.topology import Layer
from keras import backend as K
import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications import ResNet50
from keras.applications.vgg16 import VGG16
import os
from keras import backend as K
from keras.preprocessing import image
import keras.backend as K
import efficientnet.keras as efn

def get_size_vgg_feat_map(input_W, input_H):
    output_W = input_W
    output_H = input_H
    for i in range(1,6):
        output_H = np.floor(output_H/2)
        output_W = np.floor(output_W/2)

    return output_W, output_H

def rmac_regions(W, H, L):
    w = min(W,H)
    idx=0
    # region overplus per dimension
    Wd, Hd = 0, 0
    if H < W:
        Wd = idx + 1
    elif H > W:
        Hd = idx + 1

    regions = []

    for l in range(1,L+1):

        wl = np.floor(2*w/(l+1))#size of regoin
        wl2 = np.floor(wl/2 - 1)

        b = (W - wl) / (l + Wd - 1)    #basepoint
        if np.isnan(b): # for the first level
            b = 0
        cenW = np.floor(wl2 + np.arange(0,l+Wd)*b) - wl2 # center coordinates ## x in (x,y,size,size)

        b = (H-wl)/(l+Hd-1)
        if np.isnan(b): # for the first level
            b = 0
        cenH = np.floor(wl2 + np.arange(0,l+Hd)*b) - wl2 # center coordinates

        for i_ in cenH:
            for j_ in cenW:
                # R = np.array([i_, j_, wl, wl], dtype=np.int)
                R = np.array([j_, i_, wl, wl], dtype=np.int)
                 #R = np.array([j_, i_, j_+wl, i_+wl], dtype=np.int)
                if not min(R[2:]):
                    continue

                regions.append(R)

    regions = np.asarray(regions)
    return regions


class RoiPooling(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, num_rois, **kwargs):

        self.dim_ordering = K.common.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list
        self.num_rois = num_rois

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.nb_channels * self.num_outputs_per_channel

    def get_config(self):
        config = {'pool_list': self.pool_list, 'num_rois': self.num_rois}
        base_config = super(RoiPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = [w / i for i in self.pool_list]
            col_length = [h / i for i in self.pool_list]

            if self.dim_ordering == 'th':
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x + ix * col_length[pool_num]
                            x2 = x1 + col_length[pool_num]
                            y1 = y + jy * row_length[pool_num]
                            y2 = y1 + row_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')

                            new_shape = [input_shape[0], input_shape[1],
                                         y2 - y1, x2 - x1]
                            x_crop = img[:, :, y1:y2, x1:x2]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(2, 3))
                            outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x + ix * col_length[pool_num]
                            x2 = x1 + col_length[pool_num]
                            y1 = y + jy * row_length[pool_num]
                            y2 = y1 + row_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')

                            new_shape = [input_shape[0], y2 - y1,
                                         x2 - x1, input_shape[3]]
                            x_crop = img[:, y1:y2, x1:x2, :]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(1, 2))
                            outputs.append(pooled_val)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.nb_channels * self.num_outputs_per_channel))

        return final_output


def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def rmac(input_shape1, num_rois):
    
    # Load VGG16efn.EfficientNetB0
    #vgg16_model = VGG16(utils.DATA_DIR + utils.WEIGHTS_FILE, input_shape1)
    #print(len(regions))

    vgg16_model =efn.EfficientNetB1(include_top=False, weights='imagenet', input_shape=input_shape1 , pooling=None)
    
    #vgg16_model.summary()
    #print("getting wieghts")
    #print(vgg16_model.layers[-2].get_weights())
    # Regions as input
    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    # ROI pooling
    x = RoiPooling([1], num_rois)([vgg16_model.layers[-1].output, in_roi])
    
    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

    # PCA
    x = TimeDistributed(Dense(1280, name='pca',kernel_initializer='identity',bias_initializer='zeros'))(x)#,
                              #kernel_initializer='identity',
                              #bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(1280,), name='rmac')(x)

    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)

    # Define model
    model = Model([vgg16_model.input, in_roi], rmac_norm)
    #model.summary()

    #pickle_in = open("/content/drive/My Drive/drive1/paris.pickle","rb")
    #w = pickle.load(pickle_in)
   # w=np.array(w)
    #n=512
    
    
    #b=listofzeros = [0] * n
    #b=np.array(b)
    model.layers[-4]#.set_weights([w, b])#, b
    return model


def get_rmac(imag):
    
    img = imag
    IMG_SIZE=1024
    scale = IMG_SIZE / max(img.size)
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
    names=[]
    DATADIR = "/content/drive/My Drive/paris6k/" #path of paris6K dataset
    #DATADIR = "/content/drive/My Drive/queries112/" #path of queries

    CATEGORIES = ["defense","eiffel","general","invalides","louvre","moulinrouge", "museedorsay","notredame","pantheon","pompidou","sacrecoeur","triomphe"] #classes of paris6K
    #CATEGORIES = ["defense","eiffel","invalides","louvre","moulinrouge", "museedorsay","notredame","pantheon","pompidou","sacrecoeur","triomphe"]
    for category in CATEGORIES: #class by class : e.g. first class "defense"
        
        path = os.path.join(DATADIR,category)   #add the class name to the path(DATADIR) ==> /content/drive/My Drive/paris6k/defense/
        for img in (os.listdir(path)):    #take image from images in class 
            
            results=[]
            if "("  in img: #check for duplicated images
              print("double")
              continue
              
            
            try:
                imag = image.load_img(os.path.join(path,img))#add the image name to the path ==> /content/drive/My Drive/paris6k/defense/paris_defense0001.jpg and load image
                RMAC = get_rmac(imag)#get image extracted feature pathing the image to the  conveloution neural network using Rmac pooling
                learned_codes.append(RMAC))#save the extracted features of image
                print(photo_number)
                names.append(img)#save the name of the image
                print(img)
            except Exception as e: #Skip corrupted images
                print(e)
             
            print('Done!')
            photo_number=photo_number+1
            from keras import backend as K #clear RAM
            K.clear_session()
            
            #save images features and names in pickle file
            pickle_out = open("/content/drive/My Drive/learned_code_parisB1RMAC.pickle","wb")
            pickle.dump(learned_codes, pickle_out)
            pickle_out.close()
            pickle_out = open("/content/drive/My Drive/paris_namesB1RMAC.pickle","wb")
            pickle.dump(names, pickle_out)
            pickle_out.close()