# publicly available docker image 'python' on docker hub will be pulled
FROM python:3.8.2
FROM tensorflow/tensorflow:2.4.0-gpu

# Set work directory
WORKDIR home/

# Install requirements
RUN apt-get update -y && apt-get upgrade -y # && apt-get dist-upgrade -y

RUN pip install --upgrade pip && pip install \
    tensorflow-gpu==2.4.0 \
    scikit-image==0.16.2 \
    numpy \
    scipy \
    Pillow \
    cython \
    matplotlib \
    opencv-python \
    h5py \
    imgaug \
    pyyaml \
    requests \
    icecream \
    #tensorflow-probability==0.5.0 \
    keras \
    --no-cache-dir

# Copy necessary folders
RUN mkdir datasets
RUN mkdir mrcnn
RUN mkdir samples

COPY datasets/ /home/datasets
COPY mrcnn/ /home/mrcnn
COPY samples/ /home/samples

RUN ls -la *

WORKDIR /home/samples/balloon/

RUN ls -la *
RUN du -h
# Copy necessary files
COPY gpus.py .

# Run code
CMD ["python",  "gpus.py"]
#CMD ["python", "balloon.py train --dataset=../../datasets/balloon --weights=coco"]
CMD ["python", "balloon.py", "--weights", "coco"]
# ENTRYPOINT [ "python" ]
