from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
# from keras.api.layers import TFSMLayer
from keras.api.models import load_model

app = FastAPI()

URL = "http://localhost:8501/v1/models/potatoes:predict"
# MODEL = load_model("../models/potatoes/1/potatoes_1.keras")
CLASS_NAME = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']



@app.get("/")
async def home():
    return "Hey I am Alive !"

@app.get("/ping")
async def ping():
    return "Hey I am Alive !"



def read_file_as_image(data)->np.ndarray:
    img = np.array(Image.open(BytesIO(data)))
    return img

@app.post("/predict")
async def predict(file: UploadFile= File(...)):
    image =  read_file_as_image(await file.read())
    # print(image)
    image_batch = np.expand_dims(image,0)
    
    #Preparing Json Data
    json_data = {
        "instances": image_batch.tolist()
    }
    
    response = requests.post(URL,json=json_data)
    print(f"Response From TF Serving : {response.json()}")
    prediction = np.array(response.json()["predictions"][0])
    
    #prep return Response
    predicted_class = CLASS_NAME[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        "class":predicted_class,
        "confidence":float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8001)