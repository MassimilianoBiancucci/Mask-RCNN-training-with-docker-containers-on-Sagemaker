FROM maskrcnn_docker:latest

ENV SCRIPT_NAME="train.py"


# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training

# Copies the training code inside the container
COPY $SCRIPT_NAME /opt/ml/code/$SCRIPT_NAME

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM $SCRIPT_NAME

