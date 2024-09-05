import base64
from io import BytesIO
from PIL import Image

def save_base64_image(file_path, save_path):
    """
    Reads a base64-encoded image string from a file, decodes it, and saves it to the specified path.
    
    Args:
        file_path (str): The path to the file containing the base64-encoded image string.
        save_path (str): The path where the decoded image will be saved.
    """
    # Read the base64-encoded image string from the text file
    with open(file_path, 'r') as file:
        base64_image = file.read()

    # Decode the base64 string
    image_data = base64.b64decode(base64_image)

    # Open the image using Pillow
    image = Image.open(BytesIO(image_data))

    # Save the image to the specified path
    image.save(save_path)

# Example usage
if __name__ == "__main__":
    file_path = 'image_string.txt'
    save_path = 'decoded_image.png'
    save_base64_image(file_path, save_path)
    print(f"Image saved to {save_path}")