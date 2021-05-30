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
import json


class LesionBoundaryConfig(Config):
    """
    estendo la classe Config di maskrcnn (mrcnn/config.py) che contiene le 
    configurazini di default e ovverrido quelle che voglio modificare.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super().__init__()


class LesionBoundaryDataset(utils.Dataset):

    def __init__(self, imagePaths, masks_path, classNames, width=1024):
        # call the parent constructor
        super().__init__(self)

        # store the image paths and class names along with the width
        # we'll resize images to
        self.imagePaths = imagePaths
        self.masks_path = masks_path
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
        # grab the image info and derive the full annotation path
        # file path
        info = self.image_info[imageID]
        filename = info["id"].split(".")[0]

        annotPath = os.path.sep.join([self.masks_path,
                                      f"{filename}_segmentation.png"])

        # load the annotation mask and resize it, *making sure* to
        # use nearest neighbor interpolation
        annotMask = cv2.imread(annotPath)
        annotMask = cv2.split(annotMask)[0]
        annotMask = imutils.resize(annotMask,
                                   width=self.width,
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

    '''
    os.environ['SM_CHANNELS'] = '["dataset","model"]'
    os.environ['SM_CHANNEL_DATASET'] = '/opt/ml/input/data/dataset'
    os.environ['SM_CHANNEL_MODEL'] = '/opt/ml/input/data/model'   
    os.environ['SM_HPS'] = '{"NAME": "lesion", \
                             "GPU_COUNT": 1, \
                             "IMAGES_PER_GPU": 1,\
                             "CLASS_NAMES": {"1": "lesion"},\
                             "TRAINING_SPLIT": 0.8,\
                             "TRAIN_SEQ":[\
                                {"epochs": 20, "layers": "heads", "lr": 0.001},\
                                {"epochs": 40, "layers": "all", "lr": 0.0001 }\
                             ]\
                            }'
    '''

    # default env vars
    user_defined_env_vars = {"checkpoints": "/opt/ml/checkpoints",
                             "tensorboard": "/opt/ml/output/tensorboard"}

    channels = read_channels()

    dataset_path = channels['dataset']
    MODEL_PATH = os.path.sep.join([channels['model'], "mask_rcnn_coco.h5"])
    CHECKPOINTS_DIR = read_env_var("checkpoints", user_defined_env_vars["checkpoints"])
    TENSORBOARD_DIR = read_env_var("tensorboard", user_defined_env_vars["tensorboard"])
    hyperparameters = json.loads(read_env_var('SM_HPS', {}))
    
    # TODO se cambi dataset sta cosa non funziona piu'!
    images_path = os.path.sep.join([dataset_path,
                                    "ISIC2018_Task1-2_Training_Input"])
    masks_path = os.path.sep.join([dataset_path,
                                   "ISIC2018_Task1_Training_GroundTruth"])

    # initialize the amount of data to use for training
    TRAINING_SPLIT = hyperparameters['TRAINING_SPLIT']

    # grab all image paths, then randomly select indexes for both training
    # and validation
    image_paths = sorted(list(paths.list_images(images_path)))

    # TODO solo per test!!
    # image_paths = image_paths[:50]

    idxs = list(range(0, len(image_paths)))
    random.seed(42)
    random.shuffle(idxs)
    i = int(len(idxs) * TRAINING_SPLIT)

    print("training samples:" + str(i))
    print("validations samples:" + str(len(idxs) - i))

    trainIdxs = idxs[:i]
    valIdxs = idxs[i:]

    CLASS_NAMES = {
        int(k): v for k, v in hyperparameters['CLASS_NAMES'].items()
        }

    # load the training dataset
    trainDataset = LesionBoundaryDataset(image_paths, masks_path, CLASS_NAMES)
    trainDataset.load_lesions(trainIdxs)
    trainDataset.prepare()

    # load the validation dataset
    valDataset = LesionBoundaryDataset(image_paths, masks_path, CLASS_NAMES)
    valDataset.load_lesions(valIdxs)
    valDataset.prepare()

    # da mettere negli iperparametri
    GPU_COUNT = hyperparameters['GPU_COUNT']
    IMAGES_PER_GPU = hyperparameters['IMAGES_PER_GPU']

    # initialize the training configuration
    # set the number of steps per training epoch and validation cycle
    STEPS_PER_EPOCH = len(trainIdxs) // (IMAGES_PER_GPU * GPU_COUNT)
    VALIDATION_STEPS = len(valIdxs) // (IMAGES_PER_GPU * GPU_COUNT)

    # number of classes (+1 for the background)
    NUM_CLASSES = len(CLASS_NAMES) + 1

    config = LesionBoundaryConfig(
        STEPS_PER_EPOCH=STEPS_PER_EPOCH,
        VALIDATION_STEPS=VALIDATION_STEPS,
        NUM_CLASSES=NUM_CLASSES,
        **hyperparameters,
    )

    #print all config varaibles
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
                              checkpoints_dir=CHECKPOINTS_DIR,
                              tensorboard_dir=TENSORBOARD_DIR)

    # check if there is any checkpoint in the checkpoint folder
    # if there are, load the last checkpoint
    try:
        if os.listdir(model.checkpoints_dir_unique):
            MODEL_PATH = last_checkpoint_path(model.checkpoints_dir_unique, config.NAME)
    except:
        print('checkpoints folder empty...')

    # load model
    model.load_weights(MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # execute train sequence
    train_seq = hyperparameters['TRAIN_SEQ']

    print(train_seq)
    print(type(train_seq))

    for i in range(len(train_seq)):
        if model.epoch >= train_seq[i]['epochs']:
            continue
        
        model.train(trainDataset, valDataset, epochs=train_seq[i]['epochs'], 
            layers=train_seq[i]['layers'], learning_rate=train_seq[i]['lr'], augmentation=aug)

    ''' 
     OLD FASHION
    # train *just* the layer heads
    model.train(trainDataset, valDataset, epochs=hyperparameters['HEAD_TRAIN_EPOCHS'],
                layers="heads", learning_rate=config.LEARNING_RATE,
                augmentation=aug)

    # unfreeze the body of the network and train *all* layers
    model.train(trainDataset, valDataset, epochs=hyperparameters['ALL_TRAIN_EPOCHS'],
                layers="all", learning_rate=config.LEARNING_RATE / 10,
                augmentation=aug)
    '''

'''
TRAIN_SEQ hyperparameter sample

    In this notation the epochs specify a number of epoch absolute, the first object specify
    that from the epoch 0 to epoch 20 there are certain parameters, the second object specify that there is
    
    'TRAIN_SEQ':[
        {
            'epochs': 20,
            'layers': 'heads',
            'lr': 0.001,
        },
        {
            'epochs': 40,
            'layers': 'all',
            'lr': 0.0001,
        }
    ]

sample output:

1/2075 [..............................] - ETA: 21:39:54 - loss: 3.1190 - rpn_class_loss: 0.0191 - rpn_bbox_loss: 0.1407 - mrcnn_class_loss: 0.6572 - mrcnn_bbox_loss: 0.9571 -    
2/2075 [..............................] - ETA: 11:12:53 - loss: 2.8820 - rpn_class_loss: 0.0191 - rpn_bbox_loss: 0.1359 - mrcnn_class_loss: 0.5046 - mrcnn_bbox_loss: 0.9102 -    
'''

r'''
metric_definitions=[
                        {
                            "Name": "loss",
                            "Regex": "\sloss:\s(\d+.?\d*)\s-",
                        },
                        {
                            "Name": "rpn_class_loss",
                            "Regex": "\srpn_class_loss:\s(\d+.?\d*)\s-",
                        },
                        {
                            "Name": "rpn_bbox_loss",
                            "Regex": "\srpn_bbox_loss:\s(\d+.?\d*)\s-",
                        },
                        {
                            "Name": "mrcnn_class_loss",
                            "Regex": "\smrcnn_class_loss:\s(\d+.?\d*)\s-",
                        },
                        {
                            "Name": "mrcnn_bbox_loss",
                            "Regex": "\smrcnn_bbox_loss:\s(\d+.?\d*)\s-",
                        },
                    ]

'''