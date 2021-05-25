'''
Script for Mask_R-CNN training 
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 

'''
TF DEBUG LEVELS:
	0 = all messages are logged (default behavior)
	1 = INFO messages are not printed
	2 = INFO and WARNING messages are not printed
	3 = INFO, WARNING, and ERROR messages are not printed
'''

# import the necessary packages
from imgaug import augmenters as iaa
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2


# initialize the dataset path, images path, and annotations file path
#ATTENZONE MODIFICARE CON LA POSIZIONE CORRETTA SE NECESSARIO
DATASET_PATH = os.path.abspath("/root/isic2018")
#DATASET_PATH = os.path.abspath("isic2018")
IMAGES_PATH = os.path.sep.join([DATASET_PATH,
	"ISIC2018_Task1-2_Training_Input"])
MASKS_PATH = os.path.sep.join([DATASET_PATH,
	"ISIC2018_Task1_Training_GroundTruth"])

# initialize the amount of data to use for training
TRAINING_SPLIT = 0.8

# grab all image paths, then randomly select indexes for both training
# and validation
IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATH)))

IMAGE_PATHS = IMAGE_PATHS[:100]

idxs = list(range(0, len(IMAGE_PATHS)))
random.seed(42)
random.shuffle(idxs)
i = int(len(idxs) * TRAINING_SPLIT)

print("training samples:" + str(i))
print("validations samples:" + str(len(idxs) - i))

trainIdxs = idxs[:i]
valIdxs = idxs[i:]

# initialize the class names dictionary
CLASS_NAMES = {1: "lesion"}

# initialize the path to the Mask R-CNN pre-trained on COCO
COCO_PATH = "mask_rcnn_coco.h5"

# initialize the name of the directory where logs and output model
# snapshots will be stored
LOGS_AND_MODEL_DIR = "lesions_logs"

class LesionBoundaryConfig(Config):
	"""
	estendo la classe Config di maskrcnn (mrcnn/config.py) che contiene le 
	configurazini di default e ovverrido quelle che voglio modificare.
	"""	

	# give the configuration a recognizable name
	NAME = "lesion"

	# set the number of GPUs to use training along with the number of
	# images per GPU (which may have to be tuned depending on how
	# much memory your GPU has)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the number of steps per training epoch and validation cycle
	STEPS_PER_EPOCH = len(trainIdxs) // (IMAGES_PER_GPU * GPU_COUNT)
	VALIDATION_STEPS = len(valIdxs) // (IMAGES_PER_GPU * GPU_COUNT)

	# number of classes (+1 for the background)
	NUM_CLASSES = len(CLASS_NAMES) + 1


class LesionBoundaryInferenceConfig(LesionBoundaryConfig):
	# set the number of GPUs and images per GPU (which may be
	# different values than the ones used for training)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the minimum detection confidence (used to prune out false
	# positive detections)
	DETECTION_MIN_CONFIDENCE = 0.9


class LesionBoundaryDataset(utils.Dataset):
	def __init__(self, imagePaths, classNames, width=1024):
		# call the parent constructor
		super().__init__(self)

		# store the image paths and class names along with the width
		# we'll resize images to
		self.imagePaths = imagePaths
		self.classNames = classNames
		self.width = width

	def load_lesions(self, idxs):
		"""load the dataset from the disk into the dataset class

		Args:
			idxs (list of int): gli indici che determinano l'ordine 
						delle immagini nel dataset
		"""		

		# loop over all class names and add each to the 'lesion'
		# dataset
		for (classID, label) in self.classNames.items():
			self.add_class("lesion", classID, label)

		# loop over the image path indexes
		for i in idxs:
			# extract the image filename to serve as the unique
			# image ID
			imagePath = self.imagePaths[i]
			filename = imagePath.split(os.path.sep)[-1]

			# add the image to the dataset
			self.add_image("lesion", image_id=filename, path=imagePath)

	#override
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

	#override
	def load_mask(self, imageID):
		# grab the image info and derive the full annotation path
		# file path
		info = self.image_info[imageID]
		filename = info["id"].split(".")[0]
		annotPath = os.path.sep.join([MASKS_PATH,
			"{}_segmentation.png".format(filename)])

		# load the annotation mask and resize it, *making sure* to
		# use nearest neighbor interpolation
		annotMask = cv2.imread(annotPath)
		annotMask = cv2.split(annotMask)[0]
		annotMask = imutils.resize(annotMask, width=self.width,
			inter=cv2.INTER_NEAREST)
		annotMask[annotMask > 0] = 1

		# determine the number of unique class labels in the mask
		classIDs = np.unique(annotMask)

		# the class ID with value '0' is actually the background
		# which we should ignore and remove from the unique set of
		# class identifiers
		classIDs = np.delete(classIDs, [0])

		# allocate memory for our [height, width, num_instances]
		# array where each "instance" effectively has its own
		# "channel" -- since there is only one lesion per image we
		# know the number of instances is equal to 1
		masks = np.zeros((annotMask.shape[0], annotMask.shape[1], 1),
			dtype="uint8")

		# loop over the class IDs
		for (i, classID) in enumerate(classIDs):
			# construct a mask for *only* the current label
			classMask = np.zeros(annotMask.shape, dtype="uint8")
			classMask[annotMask == classID] = 1

			# store the class mask in the masks array
			masks[:, :, i] = classMask

		# return the mask array and class IDs
		return (masks.astype("bool"), classIDs.astype("int32"))

if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	
	args = vars(ap.parse_args())

	quit()

	# load the training dataset
	trainDataset = LesionBoundaryDataset(IMAGE_PATHS, CLASS_NAMES)
	trainDataset.load_lesions(trainIdxs)
	trainDataset.prepare()

	# load the validation dataset
	valDataset = LesionBoundaryDataset(IMAGE_PATHS, CLASS_NAMES)
	valDataset.load_lesions(valIdxs)
	valDataset.prepare()

	# initialize the training configuration
	config = LesionBoundaryConfig()
	config.display()

	# initialize the image augmentation process
	# fa l'argomentazione con al massimo 2 tipi di argomentazione
	aug = iaa.SomeOf((0, 2), [
		iaa.Fliplr(0.5),
		iaa.Flipud(0.5),
		iaa.Affine(rotate=(-10, 10))
	])

	# initialize the model and load the COCO weights so we can
	# perform fine-tuning
	model = modellib.MaskRCNN(mode="training", config=config,
		model_dir=LOGS_AND_MODEL_DIR) # separare log e model

	model.load_weights(COCO_PATH, by_name=True,
		exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
			"mrcnn_bbox", "mrcnn_mask"])

	# train *just* the layer heads
	model.train(trainDataset, valDataset, epochs=20,
		layers="heads", learning_rate=config.LEARNING_RATE,
		augmentation=aug)

	# unfreeze the body of the network and train *all* layers
	model.train(trainDataset, valDataset, epochs=40,
		layers="all", learning_rate=config.LEARNING_RATE / 10,
		augmentation=aug)

'''
sample output:

1/2075 [..............................] - ETA: 21:39:54 - loss: 3.1190 - rpn_class_loss: 0.0191 - rpn_bbox_loss: 0.1407 - mrcnn_class_loss: 0.6572 - mrcnn_bbox_loss: 0.9571 -    
2/2075 [..............................] - ETA: 11:12:53 - loss: 2.8820 - rpn_class_loss: 0.0191 - rpn_bbox_loss: 0.1359 - mrcnn_class_loss: 0.5046 - mrcnn_bbox_loss: 0.9102 -    
'''