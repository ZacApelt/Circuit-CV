## Imports
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import warnings
from circuit_engine import (
    preprocessing,
    ocr,
    calculate_endpoints,
    formatting_circuit,
    generating_circuit,
    generate_bboxes
)
warnings.filterwarnings("ignore")

## Configuration
# Load the image
image_path = "./circuits/cir10.png"  

# Load the YOLOv8 model
model = YOLO('runs/detect/train2/weights/best.pt')

# Select model
use_api = False

# OCR confidence threshold
ocr_threshold = 0.7


## Main pipeline
# Preprocess the image
preprocessing(image_path)

# Generate bounding boxes
components, bboxs = generate_bboxes(model, use_api, image_path)

# Find component finds
ocr_results = ocr(bboxs, ocr_threshold)

# Calculate endpoints
endpoints, all_connections = calculate_endpoints(ocr_results, components)

# Calculate circuit structure
all_coordinates, endpoint_coordinate_mapping = formatting_circuit(components, endpoints, all_connections)

# Generate final circuit
circuit_out = generating_circuit(all_coordinates, endpoint_coordinate_mapping, components, all_connections)
