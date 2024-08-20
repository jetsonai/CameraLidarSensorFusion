import cv2
import os

# Path to the image directory
image_dir = '/home/ubuntu/Desktop/sensor_fusion/infer_data/images/image_02/data/'

# Output video file name
output_video = 'save_video.mp4'

# Get a list of all image files in the directory
image_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Check if there are any images in the directory
if not image_files:
    print("No images found in the specified directory.")
else:
    # Load the first image to get the size
    first_image_path = os.path.join(image_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is often a safe codec for mp4
    video = cv2.VideoWriter(output_video, fourcc, 10, (width, height))  # 10 means 2fps

    # Add each image to the video
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping file {image_file}, unable to load.")
            continue

        video.write(image)

    # Release the VideoWriter
    video.release()
    print(f"Video saved as {output_video}")
