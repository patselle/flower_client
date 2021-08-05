#!/bin/bash

# build first image that setup tensorflow-gpu and all necessary packages. Name this containter tensorflow
echo '[RUN] cp Dockerfile_tensorflow Dockerfile'
cp Dockerfile_tensorflow Dockerfile

echo '[RUN] docker build -t tensorflow .'
docker build -t tensorflow .

# build second image tghat setups Maskrcnn, name this maskrcnn
echo '[RUN] cp Dockerfile_maskrcnn Dockerfile'
cp Dockerfile_maskrcnn Dockerfile

echo '[RUN] docker build -t maskrcnn .'
docker build -t maskrcnn .
