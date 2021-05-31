#!/usr/bin/env bash

cd ../..

docker build --tag 011827850615.dkr.ecr.eu-west-1.amazonaws.com/maskrcnn_repo_test:lesion -f ./Sagemaker_lesion_example/Docker_image_tf_std/Dockerfile .
docker push 011827850615.dkr.ecr.eu-west-1.amazonaws.com/maskrcnn_repo_test:lesion