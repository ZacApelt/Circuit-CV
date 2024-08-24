from inference_sdk import InferenceHTTPClient
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import pprint

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="TD5xEPznLTfQhznZz4pf"
)

# Perform inference
apiresults = CLIENT.infer("circuits/cir7.png", model_id="circuit-recognition/2")
class_dict = {0: 'V', 1: 'arr', 2: 'C', 3: 'i', 4: 'L', 5: 'l-', 6: 'R', 7: 'V'}


components = []
corners = []
print(apiresults)
for detection in apiresults['predictions']:
    print(detection)
    id = class_dict.get(detection.get("class_id"))
    # Extract the bounding box coordinates
    x = detection['x']
    y = detection['y']
    width = detection['width']
    height = detection['height']
    
    # Compute the bounding box coordinates
    x1 = x - width / 2
    y1 = y - height / 2
    x2 = x + width / 2
    y2 = y + height / 2
    
    # Round coordinates
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    components.append({'component': id, 'corners': [(x2,y1), (x1, y2)]})
print(components)

#pprint.pprint(result["predictions"])
"""Old
[{'component': 'V', 'corners': [(129, 282), (74, 345)]}, {'component': 'R', 'corners': [(198, 193), (127, 225)]},
 {'component': 'L', 'corners': [(446, 190), (351, 226)]}, {'component': 'R', 'corners': [(489, 313), (457, 387)]},
   {'component': 'R', 'corners': [(340, 278), (308, 353)]}, {'component': 'C', 'corners': [(246, 292), (197, 340)]},
     {'component': 'R', 'corners': [(597, 279), (565, 352)]}, {'component': 'C', 'corners': [(497, 247), (451, 286)]}]

new:
    [{'component': 'L', 'corners': [(451, 188), (350, 223)]}, {'component': 'R', 'corners': [(491, 307), (454, 392)]},
    {'component': 'R', 'corners': [(343, 267), (304, 365)]}, {'component': 'C', 'corners': [(246, 278), (196, 342)]},
        {'component': 'R', 'corners': [(599, 269), (562, 362)]}, {'component': 'V', 'corners': [(249, 277), (194, 342)]},
        {'component': 'R', 'corners': [(195, 191), (124, 231)]}, {'component': 'V', 'corners': [(131, 261), (74, 360)]},
            {'component': 'C', 'corners': [(498, 233), (451, 288)]}]
"""