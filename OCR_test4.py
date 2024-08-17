from doctr.models import ocr_predictor

# Load an image
image_path = './circuits/tut1.jpg'

# Create an OCR predictor
predictor = ocr_predictor.create_predictor()

# Perform OCR on the image
result = predictor(image_path)

# Print the extracted text
print(result)