#!/usr/bin/env bash

cd ..
docker build --tag maskrcnn_docker -f ./Dockerfile_Local/Dockerfile .