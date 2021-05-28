#!/usr/bin/env bash

#NOTE: this script should be executed form the root of the project

cd ..

docker build --tag maskrcnn_lesion_aws -f ./Sagemaker_lesion_example/Dockerfile .