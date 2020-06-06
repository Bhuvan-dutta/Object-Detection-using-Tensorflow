"""
Created on Sat Jun  6 17:04:28 2020

@author: bhuva_pxpvpbh
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import pathlib

sys.path.append(r'C:\Users\bhuva_pxpvpbh\Downloads\models-master\models-master\research')

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if tf.__version__<'1.4.0':
    raise ImportError('Please upgrade your tensorflow')

MODEL_NAME='ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE=MODEL_NAME+'.tar.gz'
DOWNLOAD_BASE=r'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT=r'C:\Users\bhuva_pxpvpbh\ssd_mobilenet_v1_coco_2017_11_17\frozen_inference_graph.pb'
PATH_TO_LABELS=os.path.join(r'C:\Users\bhuva_pxpvpbh\Downloads\models-master\models-master\research\object_detection\data\mscoco_label_map.pbtxt')

NUM_CLASSES=90

#opener=urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE+MODEL_FILE,MODEL_FILE)

tar_file=tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name=os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file,os.getcwd())


detection_graph=tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map=label_map_util.load_labelmap(PATH_TO_LABELS)

categories=label_map_util.convert_label_map_to_categories(label_map,NUM_CLASSES)

category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width,im_height)=image.size
    return np.array(image.getdata()).reshape((im_width,im_height,3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = r'C:\Users\bhuva_pxpvpbh\Downloads\models-master\models-master\research\object_detection\test_images'
#pathlib.Path(r'C:\Users\bhuva_pxpvpbh\Downloads\models-master\models-master\research\object_detection\test_images')
TEST_IMAGE_PATHS =[os.path.join(PATH_TO_TEST_IMAGES_DIR,'image{}.jpg'.format(i)) for i in range(1,5)]
IMAGE_SIZE=(12,8)

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)
  
  model_dir = pathlib.Path(model_dir)/"saved_model"
  
  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']
  
  return model


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]
  
  # Run inference
  output_dict = model(input_tensor)
  
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  
  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
  
  return output_dict


def show_inference(model, image_path):

 image_np = np.array(Image.open(image_path))

 output_dict = run_inference_for_single_image(model, image_np)
 print("output_dict:",output_dict['detection_classes'])

 vis_util.visualize_boxes_and_labels_on_image_array(
     image_np,
     output_dict['detection_boxes'],
     output_dict['detection_classes'],
     output_dict['detection_scores'],
     category_index,
     instance_masks=output_dict.get('detection_masks_reframed', None),
     use_normalized_coordinates=True,
     line_thickness=8)
    
 
 display(Image.fromarray(image_np))  
 plt.figure(figsize=(12,8))
 plt.show(image_np) 


model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

for image_path in TEST_IMAGE_PATHS:
    show_inference(detection_model, image_path)
    
    