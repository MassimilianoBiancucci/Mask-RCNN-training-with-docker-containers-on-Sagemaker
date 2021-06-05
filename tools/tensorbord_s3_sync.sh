#!/bin/sh

DEFAULTSOURCE='s3://testtflogs/logs/'
DEFAULTTARGET='.'

S3SOURCE=${1:-$DEFAULTSOURCE}
TARGET=${2:-$DEFAULTTARGET}

while [ true ]
do
   echo "update.."
   aws s3 sync $S3SOURCE $TARGET
   sleep 5
done
