# publicly available docker image 'python' on docker hub will be pulled
FROM python:3.8.2
FROM tensorflow/tensorflow:2.4.0-gpu

# Set work directory
WORKDIR home/

# Install requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    iputils-ping \
    apt-utils \
    net-tools \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



# Copy requirements.txt
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary folders
RUN mkdir datasets
RUN mkdir maskrcnn
RUN mkdir examples
RUN mkdir federated
RUN mkdir logs
RUN mkdir src


COPY datasets/ /home/datasets
COPY maskrcnn/ /home/maskrcnn
COPY examples/ /home/examples
COPY federated/ /home/federated
COPY logs/ /home/logs
COPY src/ /home/src

# Print all dirs and files
RUN ls -la *

WORKDIR /home/examples/

# Run code
# CMD ["python",  "gpus.py"]
# #CMD ["python", "balloon.py train --dataset=../../datasets/balloon --weights=coco"]
# CMD ["python", "balloon.py", "--weights", "coco"]
# # ENTRYPOINT [ "python" ]
