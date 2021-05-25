# **Mask R-CNN training with docker containers on Sagemaker**

Mask R-CNN for metal casting defects detection and instance segmentation using Keras and TensorFlow.

This project was possible thanks to the repository [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN), where we have adapted the code for instance segmentation written by matterport for work into a docker container and fine-tuned the pretrained model on COCO with our dataset, using Sagemaker.

- - -

# **Index**

1. [Overview](#overview)
1. [Dataset](#dataset)
    - [Original dataset](#original-dataset)
    - [Our dataset](#our-dataset)
1. [Docker containers](#docker-containers)
    - [Dockerfile AWS](#dockerfile-aws)
    - [Dokcerfile Local](#dockerfile-local)
1. [ECR repository](#ecr-repository)
    - [Credential configuration](#credential-configuration)
    - [Repository creation](#repository-creation)
1. [Sagemaker](#sagemaker)
    - [Introduction](#introduction)
    - [Notebook code](#notebook-code)
    - [Container code](#container-code)
1. [Results](#results)
1. [Useful links](#useful-links)

- - -

# **Overview**

The core of the project was the matterport implementation of [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) an architecture proposed by Ross Girshick et al., revisited using [Feature pyramid network](https://arxiv.org/pdf/1612.03144.pdf) as final stage and using [Resnet101](https://arxiv.org/pdf/1512.03385.pdf) as backbone.

- - -

# **Dataset**

## **Original dataset** (for image classification task)

The original dataset is an image collection of one type of casted metal product done with similar angle of view and with the objects every in front view.
The dataset was divided only by defected and not defected object, in fact it is a dataset for only image classification.
It's composed by 781 objects with defects and 519 object without defects, the same images are available in two resolution 512x512 and 300x300.
the dataset it's available on kaggle at this [link](https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

![Original dataset preview](https://github.com/MassimilianoBiancucci/Mask-RCNN-training-with-docker-containers-on-Sagemaker/blob/main/assets/Original_dataset_preview.png?raw=true)

## **Our dataset** (for segmentation task)

Our dataset start from the precedent mentioned image classification dataset, in which we have added masks for the segmentation task. The dataset was done using [Supervisely](https://app.supervise.ly/) a powerfull tool for create your own 3D 2D datasets, for object detection, semantic and instance segmentation.
The original dataset was made by 1300 images, due to time constraints we have only annotated 238 images. In our dataset structure are present 4 classes [disk, hole, chipping, deburring], the first is present in every image of the dataset, the other three classes are preset only in images with defected disks.

![Our dataset preview](https://github.com/MassimilianoBiancucci/Mask-RCNN-training-with-docker-containers-on-Sagemaker/blob/main/assets/Segmented_dataset_preview.png?raw=true)
(this image is only rappresentative then how classes are applied)

The dataset is released in [**supervisely format**](https://docs.supervise.ly/data-organization/00_ann_format_navi/04_supervisely_format_objects), where there is two way to extract objects bitmaps shown below.
The goal for the dataset preparation for training Mask R-CNN is to extract from the dataset for each image several masks, one for each istance of object that is present into this immage and that is labled. Each mask is a simple numpy array of bool or uint8 with shape (height, width), where the backgroung is indicated with zero and the mask of our istance is marked with pixel of value 1.
At the end of the process we need to obtain for each image a numpy array with shape (height, width, n_inst) where n_inst is the number of instace of every class in the image and an array with shape (n_inst) that contains the class of each instance into the 3D array of masks.

### **Mask images preparation**

The first way is to use the images into datasets/defect_segmentation_../masks_machine/ folder where each image have the same name of the original, but their color are mapped in different way, in this format each pixel represent a class, the associations between colors and classes can be found into the obj_class_to_machine_color.json file, presented below in json format.

```json
{
  "chipping":   [1, 1, 1],
  "deburring":  [2, 2, 2],
  "holes":      [3, 3, 3],
  "disk":       [4, 4, 4]
}
```

The image in this format isn't visible to human eye as can be seen below, but it is easy to manipulate with a simple script.
In this images it's shown as the istances appear completely black but if the small differences in the pixels are highlighted they beacame visibles.

![Mask preview](https://github.com/MassimilianoBiancucci/Mask-RCNN-training-with-docker-containers-on-Sagemaker/blob/main/assets/instance_estraction_from_mask/original_mask.png?raw=true)

All we need for start the process of instance separation is to extract from the mask, each pixel with the same value and place it in another blank image by changing the values of the class pixel to 255, The result will be n matrix for n class.
In this case in the original mask were only instances belonging to 2 classes so the generated mask for deburring and hole class are empty.
The next step is to separete all instances of each class, in the case of disk we have only one instance so the work is already done but in the chipping mask there are 11 instances so we need to calculate the position of each of that.

![Mask preview](https://github.com/MassimilianoBiancucci/Mask-RCNN-training-with-docker-containers-on-Sagemaker/blob/main/assets/instance_estraction_from_mask/extracted_classes_from_mask.png?raw=true)

This step isn't too hard, if you use opencv there is an helpfull function that given one matrix 2D with 0 and 255 values, automaticaly separetes each contiguous regions,
giving helpfull information about each region ([connectedComponentsWithStats](https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f)).
In the example below we can se as each detected region is marked with a cross, but it's important to note as in the disk mask was detected one more instace that shouldn't be here, maybe beacuse there is same disconnected pixel that was rilevated as another region so it's a good practice setting an area limit under which the region it's rejected. In the output of this function in addition to the center points of the regions, the areas of the regions there are one matrix with the same dimensions of the input matrix but this time the values of each region is marked with different values, similarly how the initial mask separete different classes.

![Mask preview](https://github.com/MassimilianoBiancucci/Mask-RCNN-training-with-docker-containers-on-Sagemaker/blob/main/assets/instance_estraction_from_mask/instaces_separation_from_class_mask.jpeg?raw=true)

The last step is similar to the first, where each class was sepatrated into multiple masks but at this point the only difference is that we're separating istances of the same class. So the last thing to do is to packing all this masks into one tensor with shape (h, w, n), and creating one vector of n elements where the class of each mask are stored. 

![Mask preview](https://github.com/MassimilianoBiancucci/Mask-RCNN-training-with-docker-containers-on-Sagemaker/blob/main/assets/instance_estraction_from_mask/separated_instances_of_class_1.png?raw=true)

### **Json annotations preparation**

- - -

## Docker containers

- - -

## ECR repository

- - -

## Sagemaker

- - -

## Results

- - -

## Useful links

### AWS docs

- [AWS cli configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
  
- [Docker ECR credentials configuration](https://docs.aws.amazon.com/AmazonECR/latest/userguide/common-errors-docker.html)
  
- [Pushing Docker image to ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html)

- [EC2 instance types](https://aws.amazon.com/it/ec2/instance-types/)

- [Sagemaker pricing](https://aws.amazon.com/sagemaker/pricing/)

### Sagemaker docs

- [Estimator reference](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)

- [Sagemaker API reference](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)

- [Sagemaker toolkits containers structure](https://docs.aws.amazon.com/sagemaker/latest/dg/amazon-sagemaker-toolkits.html)

- [Git of sagemaker training toolkit](https://github.com/aws/sagemaker-training-toolkit)

- [Sagemaker practical reference](https://sagemaker.readthedocs.io/en/stable/overview.html)

- [Using Docker containers with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html)

- [Use Checkpoints in Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)

- [Adapting Your Own Training Container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html)

- [Sagemaker environment variables](https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md)

### Dataset

- [Supervisely format objects](https://docs.supervise.ly/data-organization/00_ann_format_navi/04_supervisely_format_objects)

- [dataset annotation tool](https://supervise.ly/)

- [DTL (data trasformation lenguage) docs](https://docs.supervise.ly/data-manipulation/index)

- [project dataset](https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

- [configure kaggle apis](https://adityashrm21.github.io/Setting-Up-Kaggle/)

### git reference

- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

- [aws/amazon-sagemaker-examples/advanced-functionality/custom-training-container](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/custom-training-containers/script-mode-container)

- [svpino/tensorflow-object-detection-sagemaker](https://github.com/svpino/tensorflow-object-detection-sagemaker)

- [roccopietrini/TFSagemakerDetection](https://github.com/roccopietrini/TFSagemakerDetection)

- [shashankprasanna/sagemaker-spot-training](https://github.com/shashankprasanna/sagemaker-spot-training)

### Useful articles

- [guide to using Spot instances with Amazon SageMaker](https://towardsdatascience.com/a-quick-guide-to-using-spot-instances-with-amazon-sagemaker-b9cfb3a44a68)

### Jupyter docs

- [magic commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html#)

### Related papers

- [Mask R-CNN paper](https://arxiv.org/pdf/1703.06870.pdf)

- [Feature pyramid network paper](https://arxiv.org/pdf/1612.03144.pdf)

- [resnet50 paper](https://arxiv.org/pdf/1512.03385.pdf)
