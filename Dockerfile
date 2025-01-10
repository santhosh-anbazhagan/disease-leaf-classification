# Use the official TensorFlow Serving image as the base image
FROM tensorflow/serving:latest

# Copy the model directory with multiple versions to the TensorFlow Serving model directory
COPY ./saved_models/potatoes /saved_models/potatoes

# Set the environment variable to specify the model name
ENV MODEL_NAME=potatoes

# Expose the default TensorFlow Serving ports
EXPOSE 8501

# Run TensorFlow Serving and serve the model
CMD ["--model_base_path=/saved_models/potatoes", "--rest_api_port=8501"]
