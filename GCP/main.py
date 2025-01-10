from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import json

BUCKET_NAME = "abdd-efch" # Your Bucket Name
CLASS_NAME = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")

def predict(request):
    """Handles the prediction request."""
    global model
    try:
        # Load the model if not already loaded
        if model is None:
            model_path = "/tmp/potatoes_1.keras"  # Use /tmp for temporary storage
            download_blob(
                BUCKET_NAME,
                "models/potatoes_1.keras",
                model_path
            )
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")

        # Process the uploaded image
        if "file" not in request.files:
            return json.dumps({"error": "No file provided."}), 400
        
        image= request.files["file"]
        image = np.array(Image.open(image).convert("RGB").resize((256,256)))
        image = image/255
        img_array = tf.expand_dims(image,0)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAME[np.argmax(predictions[0])]
        confidence = float(round(100 * np.max(predictions[0]), 2))

        # Return the result
        return json.dumps({
            "class": predicted_class,
            "confidence": confidence
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return json.dumps({"error": str(e)}), 500
