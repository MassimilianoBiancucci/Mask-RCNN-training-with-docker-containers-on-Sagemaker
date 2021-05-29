#!/usr/bin/env bash

cd ..

docker build --tag 011827850615.dkr.ecr.eu-west-1.amazonaws.com/maskrcnn_repo_test:lesion_2 -f ./Sagemaker_lesion_example_2/Dockerfile .
docker push 011827850615.dkr.ecr.eu-west-1.amazonaws.com/maskrcnn_repo_test:lesion_2