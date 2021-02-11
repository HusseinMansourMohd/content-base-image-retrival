#feater extraction code using max/avg pooling
#INPUT:each image per each class in Paris 6K 
#OUTPUT: respective (extracted featuers + image name) for each input image 

#imported needed functions
import tensorflow as tf
import os
import numpy as np 
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.preprocessing import image
import pickle

#model specifies the convolutional neural network(EfficientNet,ResNet,Vgg16) , specifies pooling type (max , avarage)
model = tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet' ,pooling='max')


def check_size(imag):#this function can resize the image , here we will specifies the size as 1024 
    img = imag
    IMG_SIZE = 1024
    scale = IMG_SIZE / max(img.size)
    new_size =(int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    print('Original size: %s, Resized image: %s' %(str(img.size), str(new_size)))
    img = img.resize(new_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x=x/255.
    return x




learned_codes=[]
DATADIR = "/content/drive/My Drive/paris6k/" #path of paris6K dataset
#DATADIR = "/content/drive/My Drive/queries112/" #path of queries

CATEGORIES = ["defense","eiffel","general","invalides","louvre","moulinrouge", "museedorsay","notredame","pantheon","pompidou","sacrecoeur","triomphe"] #classes of paris6K
#CATEGORIES = ["defense","eiffel","invalides","louvre","moulinrouge", "museedorsay","notredame","pantheon","pompidou","sacrecoeur","triomphe"]

photo_number=0
r=0
names=[]
for category in CATEGORIES: #class by class : e.g. first class "defense" 
        
  path = os.path.join(DATADIR,category)  #add the class name to the path(DATADIR) ==> /content/drive/My Drive/paris6k/defense/
  
  for img in (os.listdir(path)): #take image from images in class 
    results=[]
    
    print(category)
    if "("  in img: #check for duplicated images
      print("double")
      continue

    try:
      imag = image.load_img(os.path.join(path,img))#add the image name to the path ==> /content/drive/My Drive/paris6k/defense/paris_defense0001.jpg and load image
      im=check_size(imag)#resize the image to 1024 and calculate the corresponding height and width
      results = model.predict(im)#pass the image to conveloution neural network to return the extracted features
      print(results)
      learned_codes.append(results)#save the extracted features of image
      print('Done!')
      photo_number=photo_number+1
      print(photo_number)
      names.append(img) #save the name of the image
      print(img)
    except Exception as e:#Skip corrupted images
      r=r+1
      photo_number=photo_number+1
      print("exception number", r)
      print(e)
      
#save images features and names in pickle file
pickle_out = open("/content/drive/My Drive/learned_code_parisB1.pickle","wb")
pickle.dump(learned_codes, pickle_out)
pickle_out.close()
pickle_out = open("/content/drive/My Drive/paris_namesB1.pickle","wb")
pickle.dump(names, pickle_out)
pickle_out.close()