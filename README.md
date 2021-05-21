# Segmentation-of-defects-in-metal-casting-products

Mask R-CNN for metal casting defects detection and instance segmentation using Keras and TensorFlow

## Useful links

### AWS docs

- [AWS cli configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
  
- [Docker ECR credentials configuration](https://docs.aws.amazon.com/AmazonECR/latest/userguide/common-errors-docker.html)
  
- [Pushing Docker image to ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html)

### Sagemaker docs

- [Esempo ufficiale aws](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/custom-training-containers/script-mode-container)

- [Git of sagemaker training toolkit](https://github.com/aws/sagemaker-training-toolkit)

- [Sagemaker practical reference](https://sagemaker.readthedocs.io/en/stable/overview.html)

    Per passare i parametri allo script `train.py` li si deve passare all'estimator e poi riprenderli come argomenti all'interno dello script.

    sul notebook:

    ```python
    # JSON encode hyperparameters
    def json_encode_hyperparameters(hyperparameters):
        return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}


    hyperparameters = json_encode_hyperparameters({"hp1": "value1", "hp2": 300, "hp3": 0.001})

    est = sagemaker.estimator.Estimator(
        container_image_uri,
        role,
        train_instance_count=1,
        train_instance_type='ml.m5.xlarge',
        base_job_name=prefix,
        hyperparameters=hyperparameters,
    )

    train_config = sagemaker.session.s3_input(
        f"s3://{bucket}/{prefix}/train/", content_type="text/csv"
    )
    val_config = sagemaker.session.s3_input(
        f"s3://{bucket}/{prefix}/val/", content_type="text/csv"
    )

    est.fit({"train": train_config, "validation": val_config})
    ```

    nel file `train.py`:

    ```python
    import argparse
    import os
    import json

    if __name__ =='__main__':

        parser = argparse.ArgumentParser()

        # sagemaker-containers passes hyperparameters as arguments
        parser.add_argument("--hp1", type=str)
        parser.add_argument("--hp2", type=int, default=50)
        parser.add_argument("--hp3", type=float, default=0.1)

        # This is a way to pass additional arguments when running as a script
        # and use sagemaker-containers defaults to set their values when not specified.
        parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
        parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])

        args = parser.parse_args()

        train(args.hp1, args.hp2, args.hp3, args.train, args.validation)
    ```

    Because the SageMaker imports your training script, you should put your training code in a main guard `(if __name__=='__main__':)`

- [Using Docker containers with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html)

- [Use Checkpoints in Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)

    For custom training containers and other frameworks

    If you are using your own training containers, training scripts, or other frameworks not listed in the previous section, you must properly set up your training script using callbacks or training APIs to save checkpoints to the local path ('/opt/ml/checkpoints') and load from the local path in your training script. SageMaker estimators can sync up with the local path and save the checkpoints to Amazon S3.

    ```python
    estimator = Estimator(
            ...
            checkpoint_s3_uri=checkpoint_s3_bucket -> s3://....,
            checkpoint_local_path=checkpoint_local_path -> "/opt/ml/checkpoints"
    )
    ```

    <span style="color:red">il checkpoint_local_path puo' essere cambiato con il path che gia' usa la rete in questo momento?</span>

- [Adapting Your Own Training Container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html)

  Create Dockerfile and Python training scripts

  il `Dockerfile` deve avere la struttura:

    ```Dockerfile
    # Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
    RUN pip3 install sagemaker-training

    # Copies the training code inside the container
    COPY train.py /opt/ml/code/train.py

    # Defines train.py as script entrypoint
    ENV SAGEMAKER_PROGRAM train.py
    ```

  - `RUN pip install sagemaker-training` – Installs SageMaker Training Toolkit that contains the common functionality necessary to create a container compatible with SageMaker.

  - `COPY train.py /opt/ml/code/train.py` – Copies the script to the location inside the container that is expected by SageMaker. The script must be located in this folder.

  - `ENV SAGEMAKER_PROGRAM train.py` – Takes your training script train.py as the entrypoint script copied in the `/opt/ml/code` folder of the container. This is the only environmental variable that you must specify when you build your own container.

  Per lanciare il docker:

  ```python
  import sagemaker
  from sagemaker import get_execution_role
  from sagemaker.estimator import Estimator

  estimator = Estimator(image_uri=image_uri --> da ECR,
                        role=get_execution_role(),
                        base_job_name='tf-custom-container-test-job',
                        instance_count=1,
                        instance_type='ml.p2.xlarge')

  # start training
  estimator.fit()
  ```

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
