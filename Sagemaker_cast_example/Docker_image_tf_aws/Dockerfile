FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-training:1.14.0-gpu-py36-cu100-ubuntu16.04

# aggiunta all'immagine dei file necessari per l'utilizzo di cudnn
#ADD cudnn-10.0-linux-x64-v7.6.3.30/cuda/include /usr/local/cuda-10.0/include/
#ADD cudnn-10.0-linux-x64-v7.6.3.30/cuda/lib64 /usr/local/cuda-10.0/lib64/

# effettuo l'installazione delle librerie tkinter necessarie per la visualizzazione delle finestre
# SOLO PER IL DEBUG NON VA AGGIUNTO AL CONTAINER DEFIINITIVO
RUN apt-get update && apt-get install -y git \
                                        tree \
                                        libgtk2.0-dev \
                                        pkg-config \
                    && rm -rf /var/lib/apt/lists/*

# cambia la cartella 
WORKDIR /root/

# copia il file contenente i requisiti del progetto
COPY requirements.txt .

# intstalla i requisiti del repo eccetto tensorflow
RUN pip install -r requirements.txt

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip install sagemaker-training

# le modifiche da qui in giu!
#--------------------------------------------------------------------

# create project folder
RUN mkdir -p /opt/ml/code
RUN mkdir -p /opt/ml/checkpoints
RUN mkdir -p /opt/ml/output/tensorboard
RUN mkdir -p /opt/ml/input/data

# cambia la cartella 
WORKDIR /opt/ml/code

# copio la cartella principale di lavoro
COPY Mask_RCNN .

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM 'cast_sagemaker.py'


