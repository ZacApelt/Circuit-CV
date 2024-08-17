import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import torch

# Load the YOLOv8 model
model = YOLO('runs/detect/train2/weights/best.pt')


# Load the image
image_path = "tut1.jpg"  # specify file location
image = Image.open(image_path).convert("RGB")
image2 = cv2.imread(image_path)
# Draw on the image

bounding_boxes = torch.empty((0, 4))

if image is not None:
    results = model.predict(source=image_path)

    #print("Results:", results)
    for detection in results[0]:
        box = detection.boxes.xyxy  # Extract the bounding box
        bounding_boxes = torch.cat((bounding_boxes, box), dim=0)
    annotated_image = results[0].plot()
    draw = ImageDraw.Draw(image)  



    # Replace each bounding box with whitespace
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        draw.rectangle([x_min, y_min, x_max, y_max], fill="white")

    #cv2.imshow("YOLOv8 Inference", image)
    image.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()