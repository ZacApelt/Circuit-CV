import cv2
import pytesseract

# Specify the path to the Tesseract executable (only needed for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load the image
image = cv2.imread('./circuits/tut1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optionally, apply some preprocessing to improve OCR accuracy
# For example, you might want to threshold the image
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform OCR
text = pytesseract.image_to_string(gray)

print(text)

# Optionally, display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
