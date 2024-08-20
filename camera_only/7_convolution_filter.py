import cv2
import numpy as np

# Path to the image file
image_path = '/home/ubuntu/Desktop/sensor_fusion/infer_data/images/image_02/data/0000000000.png'

# Load the original image in color for sharpening and smoothing
image_color = cv2.imread(image_path)

# Check if the images were loaded successfully
if image_color is None:
    print("Unable to load the image.")
else:
    # Create a sharpening filter kernel
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

    # Apply the sharpening filter
    sharpened_image = cv2.filter2D(image_color, -1, sharpening_kernel)

    # Create a 3x3 smoothing filter (averaging filter)
    smoothing_kernel = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]], np.float32)/9

    # Apply the smoothing filter
    smoothed_image = cv2.filter2D(image_color, -1, smoothing_kernel)

    # Display the original grayscale image and the processed images
    cv2.imshow('Sharpened Image', sharpened_image)
    cv2.imshow('Smoothed Image', smoothed_image)

    cv2.waitKey(0)
    # Terminate pressing 'ESC' key
    cv2.destroyAllWindows()
