## Imports
from ultralytics import YOLO
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
image_path = "./circuits/cir9.png"  

# Load the YOLOv8 model
model = YOLO('runs/detect/train2/weights/best.pt')

# Select model
use_api = True

# OCR confidence threshold
ocr_threshold = 0.99


## Main pipeline
# Preprocess the image
preprocessing(image_path)

# Generate bounding boxes
components, bboxs = generate_bboxes(use_api, model, image_path)

# Find component finds
ocr_results = ocr(bboxs, ocr_threshold)

# Calculate endpoints
endpoints, all_connections, all_coordinates, endpoint_coordinate_mapping = calculate_endpoints(ocr_results, components)

# Make it pretty
# formatting_circuit(components, endpoints, all_connections, all_coordinates, endpoint_coordinate_mapping)

# Generate final circuit
circuit_out = generating_circuit(all_coordinates, endpoint_coordinate_mapping, components, all_connections)
