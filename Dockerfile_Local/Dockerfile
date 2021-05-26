FROM tensorflow/tensorflow:1.14.0-gpu-py3

# aggiunta all'immagine dei file necessari per l'utilizzo di cudnn
ADD cudnn-10.0-linux-x64-v7.6.3.30/cuda/include /usr/local/cuda-10.0/include/
ADD cudnn-10.0-linux-x64-v7.6.3.30/cuda/lib64 /usr/local/cuda-10.0/lib64/

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

# le modifiche da qui in giu!
#--------------------------------------------------------------------

# copio la cartella principale di lavoro
COPY mask_rcnn_coco.h5 /opt/ml/input/data/model/

WORKDIR /opt/ml/

ENV TF_CPP_MIN_LOG_LEVEL 3

# start command  
CMD [ "bash" ]