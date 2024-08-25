import easyocr
import re
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Create an OCR reader object
reader = easyocr.Reader(lang_list=['en'], gpu=False)
image_path2 = './circuits/cir10.png'
image_path = './outputs/components_removed.png'
result = reader.readtext(image=image_path, text_threshold=1)
print(result)

# Load the image using OpenCV
image = cv2.imread(image_path)


# Draw bounding boxes on the image
for box in result:
	points, text, confidence = box
	points = [tuple(point) for point in points]
	cv2.polylines(image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
	cv2.putText(image, text, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the image (optional)
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(result)