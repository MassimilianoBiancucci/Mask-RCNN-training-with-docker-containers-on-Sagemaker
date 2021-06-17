'''
Script for Mask_R-CNN training 
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
# TF DEBUG LEVELS: should be before tf import
#     0 = all messages are logged (default behavior)
#     1 = INFO messages are not printed
#     2 = INFO and W  ARNING messages are not printed
#     3 = INFO, WARNING, and ERROR messages are not printed

import cv2
import random
import imutils
import argparse
import numpy as np
from imutils import paths
from mrcnn import utils
from mrcnn import visualize
from mrcnn import model as modellib
from mrcnn.sagemaker_utils import *
from mrcnn.config import Config
from imgaug import augmenters as iaa
import time

# NOTE: used in the load_mask function
# don't move this declaration.
CLASS_NAMES = {
	1 : "chipping",
	2 : "deburring",
	3 : "holes",
	4 : "disk"
}

CLASS_COLOR = {
	1 : (0.0, 1.0, 1.0), # yellow
	2 : (0.0, 1.0, 0.0), # green
	3 : (0.0, 0.0, 1.0), # blue
	4 : (1.0, 0.0, 0.0)  # red
}

################################################################
# DATASET TEST PATH
# put here your path to the test dataset for inferencevisualization
TEST_DATASET_PATH = "/home/massi/Dataset/cast_dataset_not_labled_part/remaining_part_2/img"
################################################################

################################################################
# MODEL PATH definitions, 
# put here your directoryes and your model name
checkpoints_path = "/home/massi/Progetti/repository_simone/Mask-RCNN-training-with-docker-containers-on-Sagemaker/logs/tests_polish/cast_test_polish_6/checkpoints"
MODEL = "mask_rcnn_cast_0250.h5"
MODEL_PATH = os.path.sep.join([checkpoints_path, MODEL])
################################################################

class castConfig(Config):
	"""
	Extension of Config class of the framework maskrcnn (mrcnn/config.py),
	"""

	def __init__(self, **kwargs):
		"""
		Overriding of same config variables
		and addition of others.
		"""
		self.__dict__.update(kwargs)
		super().__init__()

class castInferenceConfig(castConfig):
	
	NAME = "cast"

	MEAN_PIXEL = np.array([143.75, 143.75, 143.75])

	# set the number of GPUs and images per GPU (which may be
	# different values than the ones used for training)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the minimum detection confidence (used to prune out false
	# positive detections)
	DETECTION_MIN_CONFIDENCE = 0.7

	# Non-maximum suppression threshold for detection
	DETECTION_NMS_THRESHOLD = 0.3
	
	NUM_CLASSES = len(CLASS_NAMES) + 1




def display_inference_on_image(r, image_in, 
									show_bbox = True, show_mask = True, show_text = True,
									class_selector = -1, instance_selector = -1):

	image = image_in.copy()
	# loop over of the detected object's bounding boxes and
	# masks, drawing each as we go along
	for i in range(0, r["rois"].shape[0]):
		if instance_selector == i or instance_selector == -1:
			if r["class_ids"][i] == class_selector or class_selector == 0:
				if show_mask:
					mask = r["masks"][:, :, i]
					image = visualize.apply_mask(image, mask, CLASS_COLOR[r["class_ids"][i]], alpha=0.5)
				
				if show_bbox:
					image = visualize.draw_box(image, r["rois"][i], CLASS_COLOR[r["class_ids"][i]])

	# convert the image back to BGR so we can use OpenCV's
	# drawing functions
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	if show_text:
		# loop over the predicted scores and class labels
		for i in range(0, len(r["scores"])):
			if instance_selector == i or instance_selector == -1:
				if r["class_ids"][i] == class_selector or class_selector == 0:
					# extract the bounding box information, class ID, label,
					# and predicted probability from the results
					(startY, startX, endY, end) = r["rois"][i]
					classID = r["class_ids"][i]
					label = CLASS_NAMES[classID]
					score = r["scores"][i]

					# draw the class label and score on the image
					text = "{}: {:.4f}".format(label, score)
					y = startY - 10 if startY - 10 > 10 else startY + 10
					cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

	# resize the image so it more easily fits on our screen
	image = imutils.resize(image, width=512)
	return image


if __name__ == "__main__":

	ap = argparse.ArgumentParser()

	ap.add_argument("-i", "--images", help = "optional path to input image path to segment" )
	
	args = vars(ap.parse_args())

	# initialize the inference configuration
	config = castInferenceConfig()

	# initialize the Mask R-CNN model for inference
	model = modellib.MaskRCNN(mode="inference", config=config, checkpoints_dir=checkpoints_path)
	
	# load our trained Mask R-CNN
	model.load_weights(MODEL_PATH, by_name=True) # , exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
	
	image_paths = [f for f in os.listdir(TEST_DATASET_PATH)]
	
	cv2.namedWindow("Output",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Output", 600,600)

	cv2.namedWindow("Original",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Original", 600,600)

	#cv2.namedWindow("Comparison",cv2.WINDOW_NORMAL)
	#cv2.resizeWindow("Comparison", 1200,600)

	show_masks = True
	show_bboxs = True
	show_text = True
	class_selector = 0
	instance_selector = -1
	instance_class_selector = 0

	img_idx = 0
	while True:
	#for i in range(len(image_paths)):

		if img_idx >= len(image_paths):
			img_idx = 0

		image_path = os.path.sep.join([TEST_DATASET_PATH, image_paths[img_idx]])
		# load the input image, convert it from BGR to RGB channel
		# ordering, and resize the image
		image_original = cv2.imread(image_path)
		image = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
		image = imutils.resize(image, width=1024)

		print("start inference...")

		# perform a forward pass of the network to obtain the results
		tic = time.perf_counter()
		r = model.detect([image], verbose=1)[0]
		toc = time.perf_counter()
		print(f"Elapsed for inference: {(toc - tic)*1000:0.2f} ms")

		#print(r)

		cv2.imshow("Original", image_original)

		while(True):
			
			image_mask = display_inference_on_image(r, image, 
											show_bboxs, show_masks, show_text,
											class_selector, instance_selector)

			# show the output image
			cv2.imshow("Output", image_mask)
			
			#image_hstack = np.hstack((image_original, image_mask))
			#cv2.imshow("Comparison", image_hstack)

			key = cv2.waitKey(0)
			
			print(key)

			# if b pressed toogle bounding boxes
			if key == ord('b'):
				show_bboxs = not show_bboxs
			
			# if m pressed toogle masks
			elif key == ord('m'):
				show_masks = not show_masks
			
			# if t pressed toogle text
			elif key == ord('t'):
				show_text = not show_text

			# if c pressed selective show class
			# swipe trough classes
			elif key == ord('c'):
				class_selector = class_selector+1 if class_selector < len(CLASS_NAMES) else 0
				instance_selector = -1

			# if i pressed selective show instances
			elif key == ord('i'):
				if class_selector == 0:
					instance_selector = instance_selector+1 if instance_selector < r["rois"].shape[0] else 0
				else:
					class_instance_idxs = [i for i, instance_class in enumerate(r["class_ids"]) if instance_class == class_selector]
					print(f"class_instance_idxs: {class_instance_idxs}")
					instance_class_selector = instance_class_selector+1 if instance_class_selector < len(class_instance_idxs)-1 else 0
					print(f"instance_class_selector: {instance_class_selector}")
					if class_instance_idxs:
						instance_selector = class_instance_idxs[instance_class_selector]
						print(f"instance_selector: {instance_selector}")

			# if p pressed the previus image will be shown
			elif key == ord('p'):
				if img_idx > 0:
					img_idx -= 2
					break

			# if r pressed all settings will be resetted
			elif key == ord('r'):
				instance_selector = -1
				instance_class_selector = 0
				class_selector = 0
				show_text = True
				show_bboxs = True
				show_masks = True

			#if h is pressed help will be prompted
			elif key == ord('h'):
				print(
"""HELP:

b - if pressed all bounding boxxes are toggled

m - if pressed all masks are toogled

c - if pressed only one selected class will be showed
	if pressed again the next class will be showed
	after the last class all classes wil be restored

i - if pressed only one selected instance will be showed
	if pressed again the next instance will be showed
	after the last instance all instances wil be restored

r - if pressed restore all settings to the initial conditions

p - if pressed show the previus image

q - if pressed close all windows and the script

NOTE: all other keys scrolls to the next image""")

			# if q pressed quit
			elif key == ord('q'):
				cv2.destroyAllWindows() 
				raise SystemExit
				
			else:
				break
		
		img_idx += 1
		instance_selector = -1
		instance_class_selector = 0

	#closing all open windows 
	cv2.destroyAllWindows() 
