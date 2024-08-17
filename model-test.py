import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import torch
import OCR_test3 as ocr
import easyocr


# Load the YOLOv8 model
model = YOLO('runs/detect/train2/weights/best.pt')

# Load the image
image_path = "./circuits/cir7.png"  # specify file location 
image = Image.open(image_path).convert("RGB")

# Create an OCR reader object
reader = easyocr.Reader(['en'])

# Initialize an empty tensor for bounding boxes
component_bounding_boxes = torch.empty((0, 4))

if image is not None:
    # Get component bounding boxes using YOLO
    results = model.predict(source=image_path)

    # Extract component bounding boxes
    for detection in results[0]:
        box = detection.boxes.xyxy  # Extract the bounding box
        print("boudning boxes:\n", detection.boxes.xyxy)
        component_bounding_boxes = torch.cat((component_bounding_boxes, box), dim=0)
        

    # Create a drawing object
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
    component_removed_image.show()
    component_removed_image.save("./circuits/clean_circuit.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
