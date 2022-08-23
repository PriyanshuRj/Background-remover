import os
import cv2
import random
import string
import io as IO
import numpy as np
from skimage import io
import tensorflow as tf 
import matplotlib.pyplot as plt

upload_folder = "images"
model_folder = "models"


def image_load(image_location):
    img = io.imread(image_location)
    img = cv2.resize(img[:,:,0:3], (256,256), interpolation=cv2.INTER_AREA)
    return img

def image_save(img,image_name):
    image_location = os.path.join(upload_folder, image_name)
    plt.imsave(image_location, img)
    return image_location


def pred_fun(imgpath=None,model=None):
    imgs = io.imread(imgpath)
    imgs = cv2.resize(imgs,(256,256))
    
    img = np.array(imgs)/255
    img = img.reshape((1,)+img.shape)
    pred = model.predict(img)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2BGRA)
    prd = pred.copy()
    prd = prd.reshape(prd.shape[1:-1])

    prd[np.where(prd>0.87)] = 1
    imgs[:,:,0] = imgs[:,:,0]*prd
    imgs[:,:,1] = imgs[:,:,1]*prd
    imgs[:,:,2] = imgs[:,:,2]*prd
    print(imgs.shape)
    
    return imgs,prd

def use_model(path,given_name):
    model_path = os.path.join(model_folder, 'model6_fulldata_epoch13.json')
    weight_path = os.path.join(model_folder, 'model6_fulldata_epoch14.h5')
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(weight_path)
    image_location = os.path.join(upload_folder, path)
    print(image_location)
    image = image_load(image_location)
    image_name = 'image_' + ''.join(random.choices(
        string.ascii_uppercase + string.ascii_lowercase + string.digits, k = 2)) + given_name
    image_location = image_save(image, image_name)
    pred,mask = pred_fun(imgpath=image_location, img=None,model=model)
    pred_image_name = 'pred_'+ image_name
    pred_location = image_save(pred, pred_image_name) 

    p_map_image_name = 'mp_'+ image_name
    p_map_location = image_save(mask, p_map_image_name)  


if __name__ == "__main__" :
    use_model("self_photo.jpg","file_name.jpg")
