from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiohttp
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

app = FastAPI()

# Load the InceptionResNetV2 NSFW model
model_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
model = hub.load(model_url)

# Define the request body schema
class ImageRequest(BaseModel):
    image_url: str

# Function to download and process the image
async def process_image(image_url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch the image.")
            image_bytes = await response.read()
            image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (299, 299))  # The InceptionResNetV2 model requires this size
            return image

# Route to classify the image
@app.post("/classify/")
async def classify_image(image_request: ImageRequest):
    try:
        image = await process_image(image_request.image_url)
        image = np.expand_dims(image, axis=0)
        prediction = model(image)
        is_adult_content = tf.reduce_any(prediction["detection_class_entities"] == "Adult")
        result = {"is_adult_content": bool(is_adult_content.numpy())}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing the image.")
