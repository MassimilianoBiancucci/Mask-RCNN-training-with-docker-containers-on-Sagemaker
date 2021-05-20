FROM tensorflow/tensorflow:1.14.0-gpu-py3

# effettuo l'installazione delle librerie tkinter necessarie per la visualizzazione delle finestre
# SOLO PER IL DEBUG NON VA AGGIUNTO AL CONTAINER DEFIINITIVO
RUN apt-get update && apt-get install -y git tree libgtk2.0-dev pkg-config

# cambia la cartella 
WORKDIR /root/

# 
RUN mkdir Mask_RCNN

# cambia la cartella 
WORKDIR /root/Mask_RCNN/

#copio la cartella principale di lavoro
COPY Mask_RCNN .

# ... printing repo requirements
RUN echo "Mask_RCNN requirements:"; cat requirements.txt

# intstalla i requisiti del repo eccetto tensorflow
RUN pip install -r requirements.txt

RUN mkdir isic2018

# aggiunta all'immagine dei file necessari per l'utilizzo di cudnn
ADD cudnn-10.0-linux-x64-v7.6.3.30/cuda/include /usr/local/cuda-10.0/include/
ADD cudnn-10.0-linux-x64-v7.6.3.30/cuda/lib64 /usr/local/cuda-10.0/lib64/

CMD [ "bash" ]