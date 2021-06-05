# sagemaker workflow

in this example we will study how to run a dockerfile in AWS sagemaker.

## sagemaker overview

when training sagemaker create the following tree structure ([reference](https://docs.aws.amazon.com/sagemaker/latest/dg/amazon-sagemaker-toolkits.html)):

```bash
/opt/ml
├── input
│   ├── config
│   │   ├── hyperparameters.json
│   │   └── resourceConfig.json
│   └── data
│       └── <channel_name>
│           └── <input data>
├── model
│
├── code
│
├── output
│
└── failure
```

most of those paths can be changed in the configuration phase, but is best to follow the preconfigured structure

## Dockerfile preparation

the lines strictly required by sagemaker are:

```dockerfile
# Install sagemaker-training toolkit that contains the common functionality
# necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training

[...]

# Defines the SAGEMAKER_PROGRAM environment variable,
# this variable tell to sagemaker which is the entrypoint in the
# default code folder note that is reccomended specify the entrypoint in
# this way so sagemaker can apply all the arguments to the script
ENV SAGEMAKER_PROGRAM "train.py"
```

we also extended the tree path preconfigured by sagemaker so we created a folder for the `checkpoints` and one for the output `output`, which are the [default folders](docs/Using the SageMaker Python SDK.md).

## train script preparation

in the training script, the one specified in the `SAGEMAKER_PROGRAM` environment variable, we need to unpack the args passed by sagemaker.

The args can comprehend the hyperparameters, the dataset path and most of the [environment variables](https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md) #TODO fare docs sulle variabili d'ambiente e linkarla

they are parsed using `argparse`, the names on the parsing arguments must be the same as the one passed in the sagemaker notebook.

```python
parser = argparse.ArgumentParser()

# sagemaker-containers passes hyperparameters as arguments
parser.add_argument("--hp1", type=str, default="test_param")
parser.add_argument("--hp2", type=int, default=50)
parser.add_argument("--hp3", type=float, default=0.1)

# This is a way to pass additional arguments when running as a script
# and use sagemaker-containers defaults to set their values when not specified.
parser.add_argument("--dataset", type=str, default=os.environ["SM_CHANNEL_DATASET"])

args = parser.parse_args()

print(args.hp1)
print(args.hp2)
print(args.hp3)
print(args.dataset)
```

## sagemaker notebook

the whole notebook is structured to create an `sagemaker.estimator.Estimator` and than call the `.fit` function. more info can be found inside the notebook.
