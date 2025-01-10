from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NMAE = "santhosh-i0-bucket-tf-1"
CLASS_NAME = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    
def predict(request):
    print("I will download the model now!")
    global model
    if model is None:
        download_blob(
            BUCKET_NMAE,
            "models/potatoes_1.keras",
            "/temp/potatoes_1.keras"
        )
        model =  tf.keras.models.load_model("/temp/potatoes_1.keras")
        print("model downloaded: ",model)
    image= request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((256,256)))
    image = image/255
    img_array = tf.expand_dims(image,0)
    
    predictions = model.predict(img_array)
    print(predictions)
    
    predictions = CLASS_NAME[np.argmax(predictions[0])]
    print("Predictions : ",predictions)
    confidence = float(round(100 * np.max(predictions[0]), 2))
    return {
        "class":predictions,
        "confidence":confidence
    }
    
    
    
    