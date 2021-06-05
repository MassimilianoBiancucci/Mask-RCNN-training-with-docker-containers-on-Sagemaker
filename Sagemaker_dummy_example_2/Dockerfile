FROM python

RUN pip install directory_tree

# Install sagemaker-training toolkit that contains the common functionality 
# necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training

RUN mkdir -p /opt/ml/checkpoints/

RUN mkdir -p /opt/ml/output/data/

# create the path and copy the main script into the code folder
# note the code folder is specified by sagemaker as default folder for
# the training script
COPY train.py /opt/ml/code/train.py

# Defines the SAGEMAKER_PROGRAM environment variable,
# this variable tell to sagemaker which is the entrypoint in the
# default code folder note that is reccomended specify the entrypoint in
# this way so sagemaker can apply all the arguments to the script
ENV SAGEMAKER_PROGRAM "train.py"

# if the program is launched with CMD[] something will not work.
