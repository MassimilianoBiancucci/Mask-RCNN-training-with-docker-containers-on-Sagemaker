# Sagemaker practical reference

To pass the parameters to the training script `train.py` we need to pass them to the estimator and than take them back into the python script.

## notebook part

```python
hyperparameters = {"hp1": "value1", "hp2": 300, "hp3": 0.001}

est = sagemaker.estimator.Estimator(
    container_image_uri,
    role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    base_job_name=prefix,
    hyperparameters=hyperparameters,
)

est.fit({
    "train": f"s3://{bucket}/{prefix}/train/", 
    "validation": f"s3://{bucket}/{prefix}/val/"
    })
```

## training script part

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
