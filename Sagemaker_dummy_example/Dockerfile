FROM python

RUN pip install directory_tree

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training

RUN mkdir -p /opt/ml/checkpoints

RUN mkdir -p /opt/ml/output

# crea il percorso e copia il file
COPY train.py /opt/ml/code/train.py

# define the sagemaker target script
ENV SCRIPT_NAME="train.py"

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM $SCRIPT_NAME

