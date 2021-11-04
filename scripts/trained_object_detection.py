""" usage: trained_object_detection.py [-h] [-f FILETYPE] [-i IMAGE_PATH] [-l LABEL_PATH] [-m MODEL_PATH] [-o OUTPUT_PATH] [-s SUPPRESS_SHOWING]

Partition dataset of images into training and testing sets

optional arguments:
  -h, --help     show this help message and exit
  -i IMAGE_PATH, --IMAGE_PATH IMAGE_PATH
                 Path to the folder where the images you wish to detect objects in are stored
  -l LABEL_PATH, --LABEL_PATH LABEL_PATH
                 Path to the folder containing the .pbtxt-file containing your (converted) labels
  -m MODEL_PATH, --MODEL_PATH MODEL_PATH
                 Path to the exported, trained model with the .pb file extension
  -o OUTPUT_PATH, --OUTPUT_PATH OUTPUT_PATH
  		  Path to save the detections
  -s SUPPRESS_SHOWING, --SUPPRESS_SHOWING SUPPRESS_SHOWING
  		  Set this boolean (true/false) to (not) show the detections
"""



# IMPORTS
import argparse
import os
import pathlib
import tensorflow as tf
import time
import numpy as np
import PyQt5
import matplotlib
import matplotlib.pyplot as plt
import warnings

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
from PIL import Image



# Prerequisites and preperations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel("ERROR")           # Suppress TensorFlow logging (2)
matplotlib.use("qt5agg")
warnings.filterwarnings("ignore")   # Suppress Matplotlib warnings



# Initiate argument parser
parser = argparse.ArgumentParser(description="Detect objects in images in given folder",
                                     formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
  "-i", "--IMAGE_PATH",
  help="Path to the folder containing the images to detect objects in are stored. If not specified, the path from the tutorial will be used",
  type=str,
  default="/home/$USER/TensorFlow/workspaces/training_demo/images/images_to_detect"
  )

parser.add_argument(
  "-f", "--FILETYPE",
  help="Format of the images to detect objects in, e.g. 'jpg', 'bmp', 'png' etc. If not specified, 'jpg' is choosen.",
  type=str,
  default="jpg"
  )
  
parser.add_argument(
  "-l", "--LABEL_PATH",
  help="Path to the folder containing the label file with .pbtxt-extension. If not specified, the path from the tutorial will be used",
  type=str,
  default="/home/$USER/TensorFlow/workspaces/training_demo/annotations/label_map.pbtxt"
)

parser.add_argument(
  "-m", "--MODEL_PATH",
  help="Path to the folder containing the trained and exported model file with .pb-extension. If not specified, the path from the tutorial will be used",
  type=str,
  default="/home/$USER/TensorFlow/workspaces/training_demo/exported-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model"
)

parser.add_argument(
  "-o", "--OUTPUT_PATH",
  help="If defined, the detections will be saved at this folder. If undefined, the detections will be shown for every input picture, but will not be saved automatically",
  type=str,
  default="none"
)

parser.add_argument(
  "-s", "--SUPPRESS_SHOWING",
  help="Set this flag if you do not want or need to see the detecions pop up in seperate windows. If this flag is not set, detections will show",
  action="store_false"
)

args = parser.parse_args()



# Catch some likely problems
IMAGE_PATH = args.IMAGE_PATH.replace("/home/$USER", str(pathlib.Path.home()))
LABEL_PATH = args.LABEL_PATH.replace("/home/$USER", str(pathlib.Path.home()))
MODEL_PATH = args.MODEL_PATH.replace("/home/$USER", str(pathlib.Path.home()))
OUTPUT_PATH = args.OUTPUT_PATH.replace("/home/$USER", str(pathlib.Path.home()))
FILETYPE = args.FILETYPE.replace(".", "")


# Load the trained model:
print("Loading model...", end="")
start_time = time.time()

model = tf.saved_model.load(MODEL_PATH)
#model = model.signatures["serving_default"]

end_time = time.time()
elapsed_time = end_time - start_time
print("Done! Took {} seconds".format(elapsed_time))



# Load label map
category_index = label_map_util.create_category_index_from_labelmap(LABEL_PATH, use_display_name=True)



# Load path to images to detect objects from:
images = sorted(list(pathlib.Path(IMAGE_PATH).glob("*." + FILETYPE)))



# Start the detection
for image in images:
  
  print("Running inference for {}... ".format(image))
  start_time = time.time()
  image_name = (str(image).rsplit("/", 1)[-1]).replace(("." + FILETYPE),"")
  # Convert image to NN-usable array
  np_image = np.array(Image.open(image))
  # Convert the image to a "tensor" which TF can use
  input_tensor = tf.convert_to_tensor(np_image)
  # By Definition: The model expects an batch of images, so add an dimension (=axis) as part of it"s inputs:
  input_tensor = input_tensor[tf.newaxis,...]
  # Run the detection
  detections = model(input_tensor)
  # Get the relevant outputs (num_detections) and remove the no more needed batch dimension
  num_detections = int(detections.pop("num_detections"))
  detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
  detections["num_detections"] = num_detections

  # detection_classes should be ints.
  detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
  
  image_with_detections = np_image.copy()

  vis_utils.visualize_boxes_and_labels_on_image_array(
    image_with_detections,
    detections["detection_boxes"],
    detections["detection_classes"],
    detections["detection_scores"],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.50,
    agnostic_mode=False)


  boxes = detections["detection_boxes"]
  classes = detections["detection_classes"],
  max_boxes_to_draw = boxes.shape[0]
  scores = detections["detection_scores"]
  min_score_thresh=0.5
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      # boxes[i] is the box which will be drawn
      print ("This is a ", category_index[classes[0][i]]["name"], " at ", boxes[i])

  if args.SUPPRESS_SHOWING:
    plt.figure(image_name)
    plt.imshow(image_with_detections)

  if OUTPUT_PATH != "none":
    file_name = OUTPUT_PATH + "/" + image_name + ".jpg"
    plt.imsave(file_name, image_with_detections)
  
  end_time = time.time()
  elapsed_time = end_time - start_time
  print("Done! Took {} seconds".format(elapsed_time))
plt.show()




