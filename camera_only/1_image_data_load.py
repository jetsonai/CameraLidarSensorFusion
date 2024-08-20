import cv2

# Path to the image file
image_path = '/home/ubuntu/Desktop/sensor_fusion/infer_data/images/image_02/data/0000000000.png'

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Unable to load the image.")
else:
    # Display the image in a window
    cv2.imshow('Loaded Image', image)

    cv2.waitKey(0)

    # Terminate pressing 'ESC' key
    cv2.destroyAllWindows()



