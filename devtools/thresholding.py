from PIL import Image, ImageFilter
import numpy as np

def adaptive_threshold(image_path, output_path, block_size=51, C=20):
    """
    Load an image, convert it to grayscale, and apply adaptive thresholding.

    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the processed image.
        block_size (int): Size of the local region to calculate the threshold. Must be an odd number.
        C (int): Constant subtracted from the mean or weighted mean to calculate the threshold.
    """
    # Load the image
    image = Image.open(image_path)
    
    # Convert image to grayscale
    grayscale_image = image.convert("L")
    
    # Convert to numpy array for processing
    np_image = np.array(grayscale_image)
    
    # Apply a mean filter to calculate the local threshold
    mean_filtered_image = Image.fromarray(np_image).filter(ImageFilter.MedianFilter(size=block_size))
    np_mean_filtered_image = np.array(mean_filtered_image)
    
    # Apply adaptive thresholding
    adaptive_threshold_image = np_image > (np_mean_filtered_image - C)
    binary_image = Image.fromarray(adaptive_threshold_image * 255)  # Convert boolean array to binary image
    
    # Save the processed image
    binary_image.save(output_path)

# Example usage
input_image_path = 'circuits/frame.png'
output_image_path = 'frame_output.png'

adaptive_threshold(input_image_path, output_image_path)
