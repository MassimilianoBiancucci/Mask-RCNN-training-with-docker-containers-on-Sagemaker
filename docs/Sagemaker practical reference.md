# Sagemaker practical reference

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
