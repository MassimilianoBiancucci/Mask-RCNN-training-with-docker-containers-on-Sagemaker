#!/usr/bin/env bash

cd ..
docker build --tag maskrcnn_docker -f ./Dockerfile\ Local/Dockerfile .