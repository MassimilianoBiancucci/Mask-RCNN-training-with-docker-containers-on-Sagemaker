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
import time
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
from mrcnn.augmentation_presets import aug_presets
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from PIL import Image
import base64
import zlib
import json
import io
import imageio

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

	USE_MINI_MASK = True
	MINI_MASK_SHAPE = (512, 512)

	# Augmenters that are safe to apply to masks
	# Some, such as Affine, have settings that make them unsafe, so always
	# test your augmentation on masks
	MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes", 
						"Fliplr", "Flipud", "CropAndPad", "Affine", 
						"PiecewiseAffine", "ScaleX", "ScaleY", 
						"TranslateX", "TranslateY", "Rotate", 
						"ShearX", "ShearY", "PiecewiseAffine",
						"WithPolarWarping", "PerspectiveTransform" ]

	# SMALL MASKS DILATION PARAMETERS
	DILATE_MASKS = True
	DILATE_THERS_2 = 15000
	DILATE_THERS_1 = 500
	DILATE_ITERATIONS_2 = 10
	DILATE_ITERATIONS_1 = 10
	DILATE_KERNEL = np.ones((2, 2), 'uint8')
	
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

	def __init__(self, imagePaths, masks_path, classNames, config, width=1024):

		# call the parent constructor
		super().__init__(self)

		# store the image paths and class names along with the width
		# we'll resize images to
		self.imagePaths = imagePaths
		self.masks_path = masks_path
		self.classNames = classNames
		self.width = width
		self.config = config

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
			
			# if DILATE_MASKS it's true, too small object will be enlarged
			if self.config.DILATE_MASKS:
				if masks[:, :, i].sum() < self.config.DILATE_THERS_1:
					masks[:, :, i] = cv2.dilate(masks[:, :, i], self.config.DILATE_KERNEL, iterations = self.config.DILATE_ITERATIONS_1)
				elif masks[:, :, i].sum() < self.config.DILATE_THERS_2:
					masks[:, :, i] = cv2.dilate(masks[:, :, i], self.config.DILATE_KERNEL, iterations = self.config.DILATE_ITERATIONS_2)


			# overwrite 1px border with zeros (needed for augmentations with edge mode)
			#tick = 5 
			#masks[0:tick, 0:self.width, i] = 0
			#masks[0:self.width, 0:tick, i] = 0
			#masks[self.width-tick:self.width, 0:self.width, i] = 0
			#masks[0:self.width, self.width-tick:self.width, i] = 0

		return (masks.astype('bool'), class_idxs)


if __name__ == "__main__":

	ap = argparse.ArgumentParser()

	ap.add_argument("-m", "--mode", default="aug", help = "debug masks or augmentation. enter \"masks\" or \"aug\"" )
	
	args = vars(ap.parse_args())

	#'''
	os.environ['SM_CHANNELS'] = '["dataset","model"]'
	#os.environ['SM_CHANNEL_DATASET'] = 'datasets/cast_dataset'
	#os.environ['SM_CHANNEL_MODEL'] = 'datasets/cast_dataset'   
	os.environ['SM_HPS'] = '{"NAME": "cast", \
							 "GPU_COUNT": 1, \
							 "IMAGES_PER_GPU": 1,\
							 "TRAIN_SEQ":[\
								{"epochs": 20, "layers": "heads", "lr": 0.001},\
								{"epochs": 40, "layers": "all", "lr": 0.0001 }\
							 ]\
							}'
	#'''

	os.environ['SM_CHANNEL_DATASET'] = '/home/massi/Progetti/repository_simone/Mask-RCNN-training-with-docker-containers-on-Sagemaker/datasets/cast_dataset_polish'
	os.environ['SM_CHANNEL_MODEL'] = '/home/massi/Progetti/repository_simone/Mask-RCNN-training-with-docker-containers-on-Sagemaker/datasets/cast_dataset_polish' 

	# default env vars
	user_defined_env_vars = {"checkpoints": "/opt/ml/checkpoints",
							 "tensorboard": "/opt/ml/output/tensorboard"}

	channels = read_channels()

	dataset_path = channels['dataset']
	MODEL_PATH = os.path.sep.join([channels['model'], "mask_rcnn_coco.h5"])
	CHECKPOINTS_DIR = read_env_var("checkpoints", user_defined_env_vars["checkpoints"])
	TENSORBOARD_DIR = read_env_var("tensorboard", user_defined_env_vars["tensorboard"])

	hyperparameters = json.loads(read_env_var('SM_HPS', {}))
	
	#prova

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

	config = castConfig(
		#STEPS_PER_EPOCH=STEPS_PER_EPOCH,
		#VALIDATION_STEPS=VALIDATION_STEPS,
		NUM_CLASSES=5,
		**hyperparameters
	)

	# load the training dataset
	trainDataset = castDatasetBox(train_image_paths, train_masks_path, CLASS_NAMES, config)
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
		
		# aug = aug_presets.blend_aug().one()
		# aug = aug_presets.aritmetic_aug(sets=[0, 1, 2]).maybe_some(p=0.95, n=(1, 3))
		# aug = aug_presets.geometric_aug(sets=3).seq()
		# aug = aug_presets.color_aug().seq()
		# aug = aug_presets.preset_1()
		aug = aug_presets.preset_1()

		train_generator = modellib.data_generator(trainDataset, config, shuffle=True,
										 augmentation=aug,
										 batch_size=config.BATCH_SIZE)
		
		print(f'batch size: {config.BATCH_SIZE}')

		cv2.namedWindow("test",cv2.WINDOW_NORMAL)
		cv2.resizeWindow("test", 600,600)

		cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
		cv2.resizeWindow("mask", 600,600)
		
		start = True
		p_key = 0

		applay_mask = False
		applay_bbox = False
		update_img = False

		im_rgb = np.zeros((512, 512), dtype='uint8')
		im_rgb_masked = np.zeros((512, 512), dtype='uint8')
		r_mask = np.zeros((512, 512), dtype='uint8')

		try:	
			while(True):
			
				#(this is necessary to avoid Python kernel form crashing)
				if not start and not update_img:
					p_key = cv2.waitKey(0)
				
				# if "q" is pressed close
				if p_key == ord('q'):
					# QUIT
					raise "Quit"
				
				# if "w" is pressed 
				elif p_key == ord('w') or start or update_img:
					# swipe image

					start = False
					mask_idx = 0

					if not update_img:
						
						tic = time.perf_counter()
						train_data = next(train_generator)
						toc = time.perf_counter()

						print(f"Elapsed for generate new data: {(toc - tic)*1000:0.2f} ms")

						#print(train_data[0]) # 7
						#print(train_data[1]) # 7
						#print(train_data[0][5][0])
						#print(train_data[0][0].shape)
						#print(train_data[0][6].shape)
						
						# Using cv2.imshow() method 
						# Displaying the image 
						#im_rgb = cv2.cvtColor(train_data[0][0][0, :, :, :], cv2.COLOR_BGR2RGB)
						im_rgb = train_data[0][0][0, :, :, :]
						bitmap = train_data[0][6][0, :, :, mask_idx]
						bboxs = train_data[0][5][0][mask_idx]

						# DEBUG reconversion to original image from normalized 
						for i in range(3):	
							im_rgb[:,:,i] = im_rgb[:,:,i] + config.MEAN_PIXEL[i]

						im_rgb = im_rgb.astype('uint8')
						r_mask = np.zeros((im_rgb.shape[0], im_rgb.shape[1]), dtype='uint8')

						#print(f'r_mask shape: {r_mask.shape}')

						if any(bbox != 0 for bbox in bboxs):
							# Mask reconstruction
							bbox_w = bboxs[3] - bboxs[1]
							bbox_h = bboxs[2] - bboxs[0]
							#print(f'bbox_w: {bbox_w}')
							#print(f'bbox_h: {bbox_h}')
							r_bitmap = cv2.resize(bitmap.astype('uint8'), (bbox_w, bbox_h), interpolation=cv2.INTER_NEAREST)
							r_mask[bboxs[0]:bboxs[2], bboxs[1]:bboxs[3]] = r_bitmap*255.0

					update_img = False
					
					cv2.imshow("mask", r_mask)

					if applay_mask or applay_bbox:
						im_rgb_masked = im_rgb.copy()
						if applay_mask:
							im_rgb_masked = visualize.apply_mask(im_rgb_masked, r_mask/255, (1.0, 0.0, 0.0), alpha=0.5)
						if  applay_bbox:
							im_rgb_masked = visualize.draw_box(im_rgb_masked, bboxs, (1.0, 0.0, 0.0))
						cv2.imshow("test", im_rgb_masked)
					else:
						cv2.imshow("test", im_rgb)
					
					

				#if "e" is pressed
				elif p_key == ord('e'):
					# swipe mask
					mask_idx += 1

					bitmap = train_data[0][6][0, :, :, mask_idx]
					bboxs = train_data[0][5][0][mask_idx]

					if any(bbox != 0 for bbox in bboxs):
						
						# Mask reconstruction
						r_mask = np.zeros((im_rgb.shape[0], im_rgb.shape[1]), dtype='uint8')

						bbox_w = bboxs[3] - bboxs[1]
						bbox_h = bboxs[2] - bboxs[0]
						r_bitmap = cv2.resize(bitmap.astype('uint8'), (bbox_w, bbox_h), interpolation=cv2.INTER_NEAREST)
						r_mask[bboxs[0]:bboxs[2], bboxs[1]:bboxs[3]] = r_bitmap*255.0

						cv2.imshow("mask", r_mask)

						if applay_mask or applay_bbox:
							im_rgb_masked = im_rgb.copy()
							if applay_mask:
								im_rgb_masked = visualize.apply_mask(im_rgb_masked, r_mask/255, (1.0, 0.0, 0.0), alpha=0.5)
							if  applay_bbox:
								im_rgb_masked = visualize.draw_box(im_rgb_masked, bboxs, (1.0, 0.0, 0.0))
							cv2.imshow("test", im_rgb_masked)

					else:
						mask_idx -= 1

				#if "r" is pressed
				elif p_key == ord('r'):
					# swipe mask

					if mask_idx > 0:
						mask_idx -= 1

					bitmap = train_data[0][6][0, :, :, mask_idx]
					bboxs = train_data[0][5][0][mask_idx]
					
					# Mask reconstruction
					r_mask = np.zeros((im_rgb.shape[0], im_rgb.shape[1]), dtype='uint8')
					bbox_w = bboxs[3] - bboxs[1]
					bbox_h = bboxs[2] - bboxs[0]
					r_bitmap = cv2.resize(bitmap.astype('uint8'), (bbox_w, bbox_h), interpolation=cv2.INTER_NEAREST)
					r_mask[bboxs[0]:bboxs[2], bboxs[1]:bboxs[3]] = r_bitmap*255.0

					cv2.imshow("mask", r_mask)

					if applay_mask or applay_bbox:
						im_rgb_masked = im_rgb.copy()
						if applay_mask:
							im_rgb_masked = visualize.apply_mask(im_rgb_masked, r_mask/255, (1.0, 0.0, 0.0), alpha=0.5)
						if  applay_bbox:
							im_rgb_masked = visualize.draw_box(im_rgb_masked, bboxs, (1.0, 0.0, 0.0))
						cv2.imshow("test", im_rgb_masked)

				#if "m" pressed applay mask on original img
				elif p_key == ord("m"):
					update_img = True
					applay_mask = not applay_mask
				
				#if "m" pressed applay mask on original img
				elif p_key == ord("b"):
					update_img = True
					applay_bbox = not applay_bbox

		except Exception as e:
			
			print(e)
		
		#closing all open windows 
		cv2.destroyAllWindows() 
	
	elif args["mode"] == "demo":
		
		config = castConfig(
			#STEPS_PER_EPOCH=STEPS_PER_EPOCH,
			#VALIDATION_STEPS=VALIDATION_STEPS,
			NUM_CLASSES=5,
			**hyperparameters
		)

		# aug = aug_presets.blend_aug().one()
		# aug = aug_presets.aritmetic_aug(sets=[0, 1, 2]).maybe_some(p=0.95, n=(1, 3))
		# aug = aug_presets.geometric_aug(sets=3).seq()
		aug = aug_presets.preset_1()

		train_generator = modellib.data_generator(trainDataset, config, shuffle=True,
										 augmentation=aug,
										 batch_size=config.BATCH_SIZE)

		import imgaug as ia

		cols = 15
		rows = 15
		img_size = 256
		images_aug = []

		img_rgb_resize = np.zeros((img_size, img_size, 3), dtype='uint8')

		for i in range(cols*rows):
			
			train_data = next(train_generator)

			im_rgb = train_data[0][0][0, :, :, :]

			for i in range(3):	
					im_rgb[:,:,i] = im_rgb[:,:,i] + config.MEAN_PIXEL[i]

			img_rgb_resize = imutils.resize(im_rgb, width=img_size)

			images_aug.append(img_rgb_resize)

		# Convert cells to a grid image and save.
		result_grid_image = ia.draw_grid(images_aug, cols=cols)
		imageio.imwrite("test_img.jpg", result_grid_image)