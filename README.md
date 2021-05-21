# Segmentation-of-defects-in-metal-casting-products

Mask R-CNN for metal casting defects detection and instance segmentation using Keras and TensorFlow

## Useful links

### AWS docs

- [AWS cli configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
  
- [Docker ECR credentials configuration](https://docs.aws.amazon.com/AmazonECR/latest/userguide/common-errors-docker.html)
  
- [Pushing Docker image to ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html)

### Sagemaker docs

- [Git of sagemaker training toolkit](https://github.com/aws/sagemaker-training-toolkit)

- [Sagemaker practical reference](https://sagemaker.readthedocs.io/en/stable/overview.html)

- [Use Checkpoints in Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)

    ```text
    For custom training containers and other frameworks

    If you are using your own training containers, training scripts, or other frameworks not listed in the previous section, you must properly set up your training script using callbacks or training APIs to save checkpoints to the local path ('/opt/ml/checkpoints') and load from the local path in your training script. SageMaker estimators can sync up with the local path and save the checkpoints to Amazon S3.
    ```

    ```python
    estimator = Estimator(
            ...
            checkpoint_s3_uri=checkpoint_s3_bucket -> s3://....,
            checkpoint_local_path=checkpoint_local_path -> "/opt/ml/checkpoints"
    )
    ```

    <span style="color:red">il checkpoint_local_path puo' essere cambiato con il path che gia' usa la rete in questo momento?</span>

- [Adapting Your Own Training Container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html)

### Dataset

- [dataset annotation tool](https://supervise.ly/)

- [DTL (data trasformation lenguage) docs](https://docs.supervise.ly/data-manipulation/index)

- [project dataset](https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

- [configure kaggle apis](https://adityashrm21.github.io/Setting-Up-Kaggle/)

### git reference

- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

- [svpino/tensorflow-object-detection-sagemaker](https://github.com/svpino/tensorflow-object-detection-sagemaker)

- [roccopietrini/TFSagemakerDetection](https://github.com/roccopietrini/TFSagemakerDetection)

- [shashankprasanna/sagemaker-spot-training](https://github.com/shashankprasanna/sagemaker-spot-training)

### Useful articles

- [guide to using Spot instances with Amazon SageMaker](https://towardsdatascience.com/a-quick-guide-to-using-spot-instances-with-amazon-sagemaker-b9cfb3a44a68)

### Jpyter docs

- [magic commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html#)

### Related papers

- [maskrcnn paper](https://arxiv.org/pdf/1703.06870.pdf)

- [feature pyramid network paper](https://arxiv.org/pdf/1612.03144.pdf)

- [resnet50 paper](https://arxiv.org/pdf/1512.03385.pdf)
