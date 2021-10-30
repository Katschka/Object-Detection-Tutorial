""" usage: resize_images.py [-h] [-i IMAGE_PATH] [-x X_SIZE] [-y Y_SIZE]

Partition dataset of images into training and testing sets

optional arguments:
  -h,            --help
                 show this help message and exit
  -i IMAGE_PATH, --IMAGE_PATH IMAGE_PATH
                 Path to the folder where the images you wish to resize are stored in
  -x X_SIZE,     --X_SIZE X_SIZE
                 pixels on x-axis for absolute scaling
  -y Y_SIZE,     -- Y_SIZE Y_SIZE
                 pixels on y-axis for absolute scaling
"""



# Imports
import argparse
import pathlib
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
from skimage import data, color, io
from skimage.transform import rescale, resize



# Initiate argument parser
parser = argparse.ArgumentParser(description="Resize all images in given folder - files other than images might get you in trouble",
                                     formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
  "-i", "--IMAGE_PATH",
  help="Path to the folder containing the images to be resized",
  type=str,
  )

parser.add_argument(
  "-x", "--X_SIZE",
  help="Absolute size in pixels of the resized image on x-axis",
  type=int,
  default = 0  
)

parser.add_argument(
  "-y", "--Y_SIZE",
  help="Absolute size in pixels of the resized image on y-axis",
  type=int,
  default = 0
)
args = parser.parse_args()

# Catch some likely problems
IMAGE_PATH = args.IMAGE_PATH.replace("/home/$USER", str(pathlib.Path.home()))
X_SIZE = args.X_SIZE
Y_SIZE = args.Y_SIZE

if IMAGE_PATH[-1] != "/":
  IMAGE_PATH = IMAGE_PATH + "/"


# Load images:
images = [f for f in listdir(IMAGE_PATH) if isfile(join(IMAGE_PATH, f))]
images = [IMAGE_PATH + f for f in images]



# rescale or resize
if X_SIZE == 0 and Y_SIZE == 0:
  print("No or incompatible arguments given - please input pixel sizes, e.g. -x 1024 -y 768") 
else:
  for image in images:
    im_name = "/resized_".join(image.rsplit("/", 1))
    image = np.array(Image.open(image))
    image = resize(image, (Y_SIZE, X_SIZE), anti_aliasing=True)
    io.imsave(im_name, image)
