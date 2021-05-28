#!/usr/bin/env bash

cd ..
docker build --tag maskrcnn_docker_aws -f ./Dockerfile_Aws/Dockerfile .