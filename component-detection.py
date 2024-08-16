from inference_sdk import InferenceHTTPClient
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="TD5xEPznLTfQhznZz4pf"
)

# Perform inference
result = CLIENT.infer("tut1.jpg", model_id="circuit-recognition/2")

print(result)