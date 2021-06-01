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
from imgaug import parameters as iap
from PIL import Image
import base64
import zlib
import json
import io


# NOTE: used in the load_mask function
# don't move this declaration.
CLASS_NAMES = {
	1 : "chipping",
	2 : "deburring",
	3 : "holes",
	4 : "disk"
}

class castConfig(Config):
	"""
	Extension of Config class of the framework maskrcnn (mrcnn/config.py),
	"""

	MEAN_PIXEL = np.array([143.75, 143.75, 143.75])

	def __init__(self, **kwargs):
		"""
		Overriding of same config variables
		and addition of others.
		"""
		self.__dict__.update(kwargs)
		super().__init__()




class castDatasetBox(utils.Dataset):
	"""
	Extension of dataset utils.Dataset
	that override the import functions for the images
	and the preparation function for the annotation mask from json files.
	"""

	def __init__(self, imagePaths, masks_path, classNames, width=1024):

		# call the parent constructor
		super().__init__(self)

		# store the image paths and class names along with the width
		# we'll resize images to
		self.imagePaths = imagePaths
		self.masks_path = masks_path
		self.classNames = classNames
		self.width = width

	def load_exampls(self):
		"""
		load the dataset from the disk into the dataset class
		"""
		
		# loop over all class names and add each to the dataset
		for (classID, label) in self.classNames.items():
			self.add_class("cast", classID, label)

		# loop over the image path indexes
		for imagePath in self.imagePaths:
			# extract the image filename to serve as the unique
			# image ID
			filename = imagePath.split(os.path.sep)[-1]
			
			# add the image to the dataset
			self.add_image("cast", image_id=filename, path=imagePath)


	# defining supervisely functions
	def base64_2_mask(self, s):
		"""
		Fuction that retrive a bool matrix from a string in base 64
		
		s - (string) that contain a serialized bool matrix
		"""
		z = zlib.decompress(base64.b64decode(s))
		n = np.frombuffer(z, np.uint8)
		#n = np.fromstring(z, np.uint8) #depecated
		mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
		return mask

	def mask_2_base64(self, mask):
		"""
		Fuction for compression and serializzation of a bool matrix

		mask - 2D ndarray with type bool)
		"""
		img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
		img_pil.putpalette([0,0,0,255,255,255])
		bytes_io = io.BytesIO()
		img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
		bytes = bytes_io.getvalue()
		return base64.b64encode(zlib.compress(bytes)).decode('utf-8')

	# override
	def load_image(self, imageID):

		# grab the image path, load it, and convert it from BGR to
		# RGB color channel ordering
		p = self.image_info[imageID]["path"]
		image = cv2.imread(p)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# resize the image, preserving the aspect ratio
		image = imutils.resize(image, width=self.width)

		# return the image
		return image

	# override
	def load_mask(self, imageID):
		"""
		Override of the original mask import function, 
		this implementation extract each mask from a json file.
		"""

		# grab the image info and derive the full annotation path
		# file path
		info = self.image_info[imageID] # dict

		filename = info["id"] # es. cast_def_0_91.jpeg

		annotPath = os.path.sep.join([self.masks_path, f"{filename}.json"]) # es. cast_def_0_91.jpeg.json

		try:
			# loading annotation files
			with open(annotPath, "r") as annotJsonFile:
				annotStr = annotJsonFile.read()
		except:
			print(f"[ERROR]: error in load_mask(). file {annotPath} not found")

		# load json file as dict
		annotJson = json.loads(annotStr)

		mask_dims = [annotJson['size']['height'], annotJson['size']['width']] # extract img dimensions
		objects = annotJson['objects'] # extract instances
		n_obj = len(objects) # extract number of images

		bitmaps = [] # list of bool ndarray containing bitmaps of each instance
		origins = [] # list of coordinates of the left right angle of each bitmap into the final mask
		class_idxs = np.zeros((n_obj,), dtype="int32")  # ndarray of class idx of each bitmap

		for i in range(n_obj):

			bitmaps.append(self.base64_2_mask(objects[i]['bitmap']['data']))
			origins.append(objects[i]['bitmap']['origin'])

			# iterate over each class and extract the idx class of the i bitmap
			for class_idx, class_key in CLASS_NAMES.items():
				if class_key == objects[i]['classTitle']:
					class_idxs[i] = class_idx

		# allocate memory for our [height, width, num_instances] array
		# where each "instance" effectively has its own "channel"
		masks = np.zeros((self.width, self.width, n_obj), dtype="uint8")

		# iterate over each instance
		for i in range(n_obj):
			ox = origins[i][1]
			oy = origins[i][0]
			w = bitmaps[i].shape[0]
			h = bitmaps[i].shape[1]

			# applay the bitmap in the right place over the empty mask of the original size of the image
			mask_swap = np.zeros((mask_dims[0], mask_dims[0]), dtype="uint8")
			mask_swap[ox:ox+w, oy:oy+h] = bitmaps[i]

			# resize the image and put int the final tensor
			# NOTE: it's realy important at this point the use of cv2.INTER_NEAREST interpolation,
			masks[:, :, i] = imutils.resize(mask_swap, width=self.width, inter=cv2.INTER_NEAREST)
			
		return (masks.astype('bool'), class_idxs)


if __name__ == "__main__":

	ap = argparse.ArgumentParser()

	ap.add_argument("-m", "--mode", help = "debug masks or augmentation. enter \"masks\" or \"aug\"" )
	
	args = vars(ap.parse_args())

	#'''
	os.environ['SM_CHANNELS'] = '["dataset","model"]'
	os.environ['SM_CHANNEL_DATASET'] = '/home/massi/Progetti/Sagemaker_training_maskrcnn/datasets/cast_dataset'
	os.environ['SM_CHANNEL_MODEL'] = '/home/massi/Progetti/Sagemaker_training_maskrcnn/datasets/cast_dataset'   
	os.environ['SM_HPS'] = '{"NAME": "cast", \
							 "GPU_COUNT": 1, \
							 "IMAGES_PER_GPU": 1,\
							 "TRAIN_SEQ":[\
								{"epochs": 20, "layers": "heads", "lr": 0.001},\
								{"epochs": 40, "layers": "all", "lr": 0.0001 }\
							 ]\
							}'
	#'''

	# default env vars
	user_defined_env_vars = {"checkpoints": "/opt/ml/checkpoints",
							 "tensorboard": "/opt/ml/output/tensorboard"}

	channels = read_channels()

	dataset_path = channels['dataset']
	MODEL_PATH = os.path.sep.join([channels['model'], "mask_rcnn_coco.h5"])
	CHECKPOINTS_DIR = read_env_var("checkpoints", user_defined_env_vars["checkpoints"])
	TENSORBOARD_DIR = read_env_var("tensorboard", user_defined_env_vars["tensorboard"])

	hyperparameters = json.loads(read_env_var('SM_HPS', {}))
	


	# TRAIN DATASET DEFINITIONS -------------------------------------------------------------
	train_images_path = os.path.sep.join([dataset_path, "training", "img"])
	train_masks_path = os.path.sep.join([dataset_path, "training", "ann"])

	train_image_paths = sorted(list(paths.list_images(train_images_path)))
	#train_mask_paths = sorted(list(paths.list_images(train_masks_path)))

	train_ds_len = len(train_image_paths)
	# ---------------------------------------------------------------------------------------

	# VALID DATASET DEFINITIONS -------------------------------------------------------------
	val_images_path = os.path.sep.join([dataset_path, "validation", "img"])
	val_masks_path = os.path.sep.join([dataset_path, "validation", "ann"])

	val_image_paths = sorted(list(paths.list_images(val_images_path)))
	#val_mask_paths = sorted(list(paths.list_images(val_masks_path)))

	val_ds_len = len(val_image_paths)
	# ---------------------------------------------------------------------------------------

	# load the training dataset
	trainDataset = castDatasetBox(train_image_paths, train_masks_path, CLASS_NAMES)
	trainDataset.load_exampls()
	trainDataset.prepare()
	
	if args["mode"] == "mask":
		# determine a sample of training image indexes and loop over
		# them
		for i in trainDataset.image_ids:
			# load the image and masks for the sampled image
			print("[INFO] investigating image index: {}".format(i))
			image = trainDataset.load_image(i)
			(masks, classIDs) = trainDataset.load_mask(i)

			# show the image spatial dimensions which is HxWxC
			print("[INFO] image shape: {}".format(image.shape))

			# show the masks shape which should have the same width and
			# height of the images but the third dimension should be
			# equal to the total number of instances in the image itself
			print("[INFO] masks shape: {}".format(masks.shape))

			# show the length of the class IDs list along with the values
			# inside the list -- the length of the list should be equal
			# to the number of instances dimension in the 'masks' array
			print("[INFO] class IDs length: {}".format(len(classIDs)))
			print("[INFO] class IDs: {}".format(classIDs))

			# visualize the masks for the current image
			visualize.display_top_masks(image, masks, classIDs,
				trainDataset.class_names)
	
	elif args["mode"] == "aug":
		
		config = castConfig(
			#STEPS_PER_EPOCH=STEPS_PER_EPOCH,
			#VALIDATION_STEPS=VALIDATION_STEPS,
			NUM_CLASSES=5,
			**hyperparameters,
		)

		"""
		# initialize the image augmentation process
		# fa l'argomentazione con al massimo 2 tipi di argomentazione
		aug = iaa.SomeOf((0, 2), [
			iaa.Fliplr(0.5),
			iaa.Flipud(0.5),
			iaa.Affine(rotate=(-25, 25)),
		])
		"""

		aug = iaa.Sequential([
				iaa.SomeOf((0, 5), [
						iaa.Fliplr(0.5), # horizontaly flip with probability
						iaa.Flipud(0.5), # vertical flip with probability
						iaa.Affine(	# geometric modification
							rotate=(-25, 25), # rotation between interval (degrees)
							mode="edge" # filler type (new pixels are generated based on edge pixels)
						),
						iaa.Affine(	# geometric modification
							shear={ # simulate angled view 
								"y": (-25, 25) # interval in degrees along y axis
							},
							mode="edge" # filler type (new pixels are generated based on edge pixels)
						),
						iaa.Affine(	# geometric modification
							shear={ # simulate angled view of given interval in degrees
								"x": (-25, 25) # interval in degrees along x axis
							},
							mode="edge" # filler type (new pixels are generated based on edge pixels)
						),
					],
					random_order=True
				),
				iaa.SomeOf((0, 1), [
						iaa.Affine( # geometric modification
							scale=(1.0, 1.3) # scale immage from 100% to 130% 
						)
					]
				)
			]
		)

		train_generator = modellib.data_generator(trainDataset, config, shuffle=True,
                                         augmentation=aug,
                                         batch_size=config.BATCH_SIZE)
		
		cv2.namedWindow("test",cv2.WINDOW_NORMAL)
		cv2.resizeWindow("test", 600,600)

		cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
		cv2.resizeWindow("mask", 600,600)

		mask_idx = 0

		try:
			
			while(True):
			
				#(this is necessary to avoid Python kernel form crashing)
				p_key = cv2.waitKey(0)

				# if "q" is pressed
				if p_key == ord('q'):
					# QUIT
					raise "Quit"
				
				# if "w" is pressed
				elif p_key == ord('w'):
					# swipe image

					mask_idx = 0

					train_data = next(train_generator)
			
					#print(train_data[0]) # 7
					#print(train_data[1]) # 7
					#print(train_data[0][0].shape)
					#print(train_data[0][6].shape)

					# Using cv2.imshow() method 
					# Displaying the image 
					#im_rgb = cv2.cvtColor(train_data[0][0][0, :, :, :], cv2.COLOR_BGR2RGB)
					im_rgb = train_data[0][0][0, :, :, :]
					mask = train_data[0][6][0, :, :, mask_idx]

					#DEBUG reconversion to original image
					for i in range(3):	
						min = - config.MEAN_PIXEL[i]
						max = 255.0 - config.MEAN_PIXEL[i]
						im_rgb[:,:,i] = ((im_rgb[:,:,i] - min)/(max - min)) * 255.0

					cv2.imshow("test", im_rgb.astype('uint8'))
					cv2.imshow("mask", mask.astype('uint8')*255)

				#if "e" is pressed
				elif p_key == ord('e'):
					# swipe mask
					mask_idx += 1
					mask = train_data[0][6][0, :, :, mask_idx]
					cv2.imshow("mask", mask.astype('uint8')*255)


		except Exception as e:
			
			print(e)
		
		#closing all open windows 
		cv2.destroyAllWindows() 
			
