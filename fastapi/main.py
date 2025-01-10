from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
# from keras.api.layers import TFSMLayer
from keras.api.models import load_model

app = FastAPI()

# # Load the TensorFlow SavedModel
# layer = TFSMLayer('./models/2', call_endpoint='potatoes')

# # Example usage in a model
# from keras import Sequential
# model = Sequential([layer])

MODEL = load_model("../models/potatoes/1/potatoes_1.keras")
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
    # print(image_batch.tolist())
    prediction = MODEL.predict(image_batch)
    index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    print(f"prediction : {prediction}")
    return {
        "result":CLASS_NAME[index],
        "confidence":float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8001)