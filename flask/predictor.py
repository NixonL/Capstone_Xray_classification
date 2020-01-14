import base64
import tensorflow.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import numpy as np
import io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

import coremltools

from keras.layers import DepthwiseConv2D, ReLU
from pathlib import Path
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope


session = tensorflow.keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)

from tensorflow.keras.models import model_from_json


global graph
graph =tf.get_default_graph()

# Dictionaries
classification={0:'ELBOW', 1:'FINGER', 2:'FOREARM',3:'HAND',4:'HUMERUS',5:'SHOULDER',6:'WRIST'}
abnormality={0:'negative', 1:'positive'}

# Load the models
def get_model():
    global anatomy_model
    global elbow_model
    global finger_model
    global forearm_model
    global hand_model
    global humerus_model
    global shoulder_model
    global wrist_model
   
    
    # Anatomy Classification model 
    with open('h5/anatomy_mobilef2.json', 'r') as f:
        anatomy_model = model_from_json(f.read())
    anatomy_model.load_weights('h5/anatomy_mobilef2.h5')
    
    
 
    # Elbow model 
    with open('h5/XR_ELBOW_mf.json', 'r') as f:
        elbow_model = model_from_json(f.read())
    elbow_model.load_weights('h5/XR_ELBOW_mf.h5')
    
    
  
    # Finger model
    with open('h5/XR_FINGER_mf.json', 'r') as f:
        finger_model = model_from_json(f.read())
    finger_model.load_weights('h5/XR_FINGER_mf.h5')
    
    
   
    # Forearm model
    with open('h5/XR_FOREARM_mf.json', 'r') as f:
        forearm_model = model_from_json(f.read())
    forearm_model.load_weights('h5/XR_FOREARM_mf.h5')
    
    

    # Hand model
    with open('h5/XR_HAND_mf.json', 'r') as f:
        hand_model = model_from_json(f.read())
    hand_model.load_weights('h5/XR_HAND_mf.h5')
    
    
   
    # Humerus model
    with open('h5/XR_HUMERUS_v2.json', 'r') as f:
        humerus_model = model_from_json(f.read())
    humerus_model.load_weights('h5/XR_HUMERUS_v2.h5')
    
    
    
    # Shoulder model
    with open('h5/XR_SHOULDER.json', 'r') as f:
        shoulder_model = model_from_json(f.read())
    shoulder_model.load_weights('h5/XR_SHOULDER.h5')
    
    
 
    # Wrist model
    with open('h5/XR_WRIST_v2.json', 'r') as f:
        wrist_model = model_from_json(f.read())
    wrist_model.load_weights('h5/XR_WRIST_v2.h5')
    
    print('Ready!!')

    
get_model()

# Prepping input images
def prepare_image(file):
    img = file
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tensorflow.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# Flask API
app = Flask(__name__)

@app.route("/predict",methods=['post','get'])

# Only png images
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded)) 
    
    # First the image goes through the anatomy classifier
    prepped_image = prepare_image(image)
    with session.as_default():
        with graph.as_default():
            predictions = anatomy_model.predict(prepped_image)
    
    elbow = predictions[0][0]
    finger = predictions[0][1]
    forearm = predictions[0][2]
    hand = predictions[0][3]
    humerus = predictions[0][4]
    shoulder = predictions[0][5]
    wrist = predictions[0][6]
    part_pred = classification[np.argmax(predictions, axis=1)[0]]
    
    # After the anatomy has been classified the image 
    # will go into the corresponding model
    if part_pred == 'ELBOW': 
        prepped_image = prepare_image(image)
        with session.as_default():
            with graph.as_default():
                predictions = elbow_model.predict(prepped_image)
    if part_pred == 'FINGER': 
        prepped_image = prepare_image(image)
        with session.as_default():
            with graph.as_default():
                predictions = finger_model.predict(prepped_image)
    if part_pred == 'FOREARM': 
        prepped_image = prepare_image(image)
        with session.as_default():
            with graph.as_default():
                predictions = forearm_model.predict(prepped_image)
    if part_pred == 'HAND': 
        prepped_image = prepare_image(image)
        with session.as_default():
            with graph.as_default():
                predictions = hand_model.predict(prepped_image)
    if part_pred == 'HUMERUS': 
        prepped_image = prepare_image(image)
        with session.as_default():
            with graph.as_default():
                predictions = humerus_model.predict(prepped_image)
    if part_pred == 'SHOULDER': 
        prepped_image = prepare_image(image)
        with session.as_default():
            with graph.as_default():
                predictions = shoulder_model.predict(prepped_image)
    if part_pred == 'WRIST': 
        prepped_image = prepare_image(image)
        with session.as_default():
            with graph.as_default():
                predictions = wrist_model.predict(prepped_image)  
    neg = predictions[0][0]
    pos = predictions[0][1]
    pred = abnormality[np.argmax(predictions, axis=1)[0]]
        
    # Post the result
    response = {
        'prediction':{
                'elbow': round(elbow.item(), 4),
                'finger': round(finger.item(), 4),
                'forearm': round(forearm.item(), 4),
                'hand': round(hand.item(), 4),
                'humerus': round(humerus.item(), 4),
                'shoulder': round(shoulder.item(), 4),
                'wrist': round(wrist.item(), 4),                
                'neg': round(neg.item(), 4),
                'pos': round(pos.item(), 4),
                'part': part_pred,
                'pred': pred
                
                
        }
    }
        
        
    return jsonify(response)