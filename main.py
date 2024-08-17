import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import torch
import OCR_test3 as ocr
import easyocr
import numpy as np
from collections import deque
import pprint

# Load the YOLOv8 model
model = YOLO('runs/detect/train2/weights/best.pt')

# Load the image
image_path = "./circuits/cir7.png"  # specify file location 
image = Image.open(image_path).convert("RGB")

# Create an OCR reader object
reader = easyocr.Reader(['en'])

# Initialize an empty tensor for bounding boxes
component_bounding_boxes = torch.empty((0, 4))
components = []

if image is not None:
    # Get component bounding boxes using YOLO
    results = model.predict(source=image_path)
    # Extract component bounding boxes
    class_dict = {0: 'V', 1: 'arr', 2: 'C', 3: 'i', 4: 'L', 5: 'l-', 6: 'R', 7: 'C'}
    
    for detection in results[0]:
        id = class_dict[int(detection.boxes.cls.item())]
        
        box = detection.boxes.xyxy  # Extract the bounding box
        component_bounding_boxes = torch.cat((component_bounding_boxes, box), dim=0)

        # Convert to a list of coordinates
        x1, y1, x2, y2 = box[0]
        corners = [
            #(round(x1.item()), round(y1.item())),
            (round(x2.item()), round(y1.item())),
            #(round(x2.item()), round(y2.item())),
            (round(x1.item()), round(y2.item()))]

        components.append({'component': id, 'corners': corners})
    # Create a drawing object
    pprint.pprint(components)
    draw = ImageDraw.Draw(image)  

    # Replace each component bounding box with white space
    for box in component_bounding_boxes:
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        draw.rectangle([x_min, y_min, x_max, y_max], fill="white")

    # Save the intermediate image after component removal
    annotated_image = results[0].plot()

    # Display the annotated image
    cv2.imshow("YOLOv8 Inference", annotated_image)
    image.save("output_components_removed.png")

    # Load the image with components removed
    component_removed_image = Image.open("output_components_removed.png").convert("RGB")
    #component_removed_image = reader.readtext("output_components_removed.png")
    draw2 = ImageDraw.Draw(component_removed_image) 

    # Perform OCR on the image
    ocr_results = reader.readtext("output_components_removed.png")
    classified_results = ocr.classify(ocr_results)

    # Initialize an empty tensor for OCR bounding boxes
    ocr_bounding_boxes = torch.empty((0, 4))

    # Extract OCR bounding boxes from the classified results
    for component in classified_results:
        corner_box = component["corners"]
        
        # Convert corner format to xyxy format
        x_min = min([point[0] for point in corner_box])
        y_min = min([point[1] for point in corner_box])
        x_max = max([point[0] for point in corner_box])
        y_max = max([point[1] for point in corner_box])

        # Create a tensor in xyxy format
        corner_box_tensor = torch.tensor([[x_min, y_min, x_max, y_max]])

        # Concatenate the corner box tensor to the OCR bounding boxes tensor
        ocr_bounding_boxes = torch.cat((ocr_bounding_boxes, corner_box_tensor), dim=0)

    # Replace each OCR bounding box with white space
    for box in ocr_bounding_boxes:
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        draw2.rectangle([x_min, y_min, x_max, y_max], fill="white")

    # Show the final image with all bounding boxes removed
    #component_removed_image.show()
    component_removed_image.save("./circuits/clean_circuit.png")

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Graph processing -----------------------------------------------
# Load the binary image
image = cv2.imread('./circuits/clean_circuit.png', 0)
# Invert the image if needed (now you have white lines on a black background)
binary_image = cv2.bitwise_not(image)
thinned = cv2.ximgproc.thinning(binary_image)


# find the end points
id = 0
endpoints = []
for i in range(1, thinned.shape[0] - 1):
    for j in range(1, thinned.shape[1] - 1):
        if thinned[i, j] == 255:
            # if a white pixel
            neighbours = thinned[i-1:i+2, j-1:j+2]
            if np.sum(neighbours) == 255*2:
                endpoints.append({"id": id, "pos":(j, i)})
                # put a red pixel at the endpoint
                cv2.circle(image, (j,i), 5, (0,0,255), 2)
                # put id next to the endpoint
                cv2.putText(image, str(id), (j+10, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                id += 1
            
#print(endpoints)

# connect the end points
def find_connected_endpoints(skeleton, endpoints, start_id):
    visited = np.zeros_like(skeleton, dtype=bool)
    connections = [start_id]
    start_pos = endpoints[start_id]['pos']

    queue = deque([start_pos])
    visited[start_pos[1], start_pos[0]] = True# Mark start position as visited
    while queue:
        x, y = queue.popleft()

        # Check 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                if 0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0]:
                    if skeleton[ny, nx] == 255 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

                        # Check if the current point is an endpoint
                        for endpoint in endpoints:
                            if endpoint['pos'] == (nx, ny) and endpoint['id'] != start_id:
                                connections.append(endpoint['id'])
    return connections

# find the max id of the endpoints
max_id = max([endpoint['id'] for endpoint in endpoints])

all_connections = []
all_connections.append(find_connected_endpoints(thinned, endpoints, 0))

# Iterate over each endpoint ID
for i in range(max_id):
    # Check if the endpoint ID is already in any connection list
    if not any(i in connection_list for connection_list in all_connections):
        # Find connected endpoints for the current ID
        connected_endpoints = find_connected_endpoints(thinned, endpoints, i)
        # Append the list of connected endpoints to all_connections
        all_connections.append(connected_endpoints)

#print(all_connections)

cv2.imshow('Endpoints', thinned)
cv2.imshow('Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
