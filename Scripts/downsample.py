import cv2
import glob
import os
import tqdm

IMAGE_DIR = "../COLMAP Projects/OIVIO Tunnel/images/*"
# Makes sure downsamples folder exists (cv2 doesn't create one)
DOWNSAMPLE_DIR = "../COLMAP Projects/OIVIO Tunnel/downsamples/"

DOWNSAMPLE_SCALE_FACTOR = 10

images = glob.glob(IMAGE_DIR)  # Gets list of all images in directory

for img_path in tqdm.tqdm(images):
    image = cv2.imread(img_path)
    # Join downsampled directory with image filename
    downsample_path = os.path.join(DOWNSAMPLE_DIR, os.path.basename(img_path))

    
    # Divide width, height by scale factor
    downsample_size = (image.shape[1] // DOWNSAMPLE_SCALE_FACTOR,
                       image.shape[0] // DOWNSAMPLE_SCALE_FACTOR,)
    
    image_downsample = cv2.resize(image, downsample_size)

    # Save downsampled image
    cv2.imwrite(downsample_path, image_downsample)
