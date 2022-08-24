import os
import re
import cv2
import random
import string
import io as IO
import numpy as np
from skimage import io
import tensorflow as tf 
import zipfile
import matplotlib.pyplot as plt
import os
from pathlib import Path
from flask import Flask, render_template, request, Response, jsonify, send_file
upload_folder = "images"
model_folder = "models"
app = Flask(__name__)


# load json and create model
model_path = os.path.join(model_folder, 'model.json')
weight_path = os.path.join(model_folder, 'model.h5')

json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights(weight_path)
def image_load(image_location):
    img = io.imread(image_location)
    img = cv2.resize(img[:,:,0:3], (256,256), interpolation=cv2.INTER_AREA)
    return img

def image_save(img,image_name):
    image_location = os.path.join(upload_folder, image_name)
    plt.imsave(image_location, img)
    return image_location


def pred_fun(imgpath=None):
    imgs = io.imread(imgpath)
    imgs = cv2.resize(imgs,(256,256))
    
    img = np.array(imgs)/255
    img = img.reshape((1,)+img.shape)
    pred = model.predict(img)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2BGRA)
    prd = pred.copy()
    prd = prd.reshape(prd.shape[1:-1])

    prd[np.where(prd>0.80)] = 1
    imgs[:,:,0] = imgs[:,:,0]*prd
    imgs[:,:,0][np.where(prd!=1)] = 255

    imgs[:,:,1] = imgs[:,:,1]*prd
    imgs[:,:,1][np.where(prd!=1)] = 255

    imgs[:,:,2] = imgs[:,:,2]*prd
    imgs[:,:,2][np.where(prd!=1)] = 255
    # imgs[:,:,3] = imgs[np.where(prd!=1)] = 1
    print(imgs.shape)
    
    return imgs,prd

app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']

@app.route("/", methods= ['POST', 'GET'])
def hello():
    return render_template("index.html")
@app.route("/zip", methods= ['POST', 'GET'])
def upload_file():
    
    if request.method == "POST":
        input_image_file = request.files["image"]
        if input_image_file:
            file_ext = os.path.splitext(input_image_file.filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return jsonify({'status': 404, 'message' : "THe uploaded file is not an supported image format."})
            else : 
                image_location = os.path.join(upload_folder, input_image_file.filename)
                input_image_file.save(image_location)

                imgs = image_load(image_location)
                imgs_name = 'new_' + ''.join(random.choices(
                    string.ascii_uppercase + string.ascii_lowercase + string.digits, k = 2)) + input_image_file.filename
                imgs_location = image_save(imgs, imgs_name)

                pred,mask = pred_fun(imgpath= imgs_location)
                pred_image_name = 'pred_'+ imgs_name
                pred_location = image_save(pred, pred_image_name) 
                    
                zipf = zipfile.ZipFile('Name.zip','w', zipfile.ZIP_DEFLATED)
                for root,dirs, files in os.walk('images/'):
                    for file in files:
                        zipf.write('images/'+file)
                zipf.close()
                p = Path('images').glob('**/*')
                for i in  p:
                    os.remove(i)
                return send_file('Name.zip',
                        mimetype = 'zip')

         
        

        else:
            return jsonify({"status":"Failed","message": "Please upload an image."})
    

@app.route("/imgs", methods= ['POST', 'GET'])
def upload_file2():
    p = Path('images').glob('**/*')
    for i in  p:
        os.remove(i)
    if request.method == "POST":
        input_image_file = request.files["image"]
        if input_image_file:
            file_ext = os.path.splitext(input_image_file.filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return jsonify({'status': 404, 'message' : "THe uploaded file is not an supported image format."})
            else : 
                image_location = os.path.join(upload_folder, input_image_file.filename)
                input_image_file.save(image_location)

                imgs = image_load(image_location)
                imgs_name = 'new_' + ''.join(random.choices(
                    string.ascii_uppercase + string.ascii_lowercase + string.digits, k = 2)) + input_image_file.filename
                imgs_location = image_save(imgs, imgs_name)

                pred,mask = pred_fun(imgpath= imgs_location)
                pred_image_name = 'pred_'+ imgs_name
                pred_location = image_save(pred, pred_image_name)            
                return send_file(pred_location,
                        mimetype = 'jpg')

         
        

        else:
            return jsonify({"status":"Failed","message": "Please upload an image."})
    



if __name__ == "__main__":
    app.run()
