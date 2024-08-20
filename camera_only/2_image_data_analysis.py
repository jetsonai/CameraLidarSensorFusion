import cv2
import numpy as np

# Path to the image file
image_path = '/home/ubuntu/Desktop/sensor_fusion/infer_data/images/image_02/data/0000000000.png'

# Load the image
image = cv2.imread(image_path)

# Check Data Type
print("Image data type", type(image))

# Check if the image was loaded successfully
if image is None:
    print("Unable to load the image.")
else:
    # Split the image into Blue, Green, and Red channels
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Create an image with only the red channel
    image_red = np.zeros_like(image)
    image_red[:, :, 2] = red_channel

    # Create an image with only the green channel
    image_green = np.zeros_like(image)
    image_green[:, :, 1] = green_channel

    # Create an image with only the blue channel
    image_blue = np.zeros_like(image)
    image_blue[:, :, 0] = blue_channel

    # Convert the original image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display each channel in its own window
    cv2.imshow('Red Channel', image_red)
    cv2.imshow('Green Channel', image_green)
    cv2.imshow('Blue Channel', image_blue)
    cv2.imshow('Gray Image', image_gray)

    cv2.waitKey(0)
    # Terminate pressing 'ESC' key
    cv2.destroyAllWindows()