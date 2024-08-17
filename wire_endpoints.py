import cv2
import numpy as np

# Load the binary image
image = cv2.imread('./circuits/tut1_blank.png', 0)

# Invert the image if needed (now you have white lines on a black background)
binary_image = cv2.bitwise_not(image)

thinned = cv2.ximgproc.thinning(binary_image)

cv2.imshow('Endpoints', thinned)
cv2.waitKey(0)
cv2.destroyAllWindows()