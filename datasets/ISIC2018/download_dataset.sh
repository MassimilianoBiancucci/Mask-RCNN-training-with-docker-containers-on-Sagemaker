
set -e

wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip

wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip

mkdir dataset

unzip -d dataset/ *.zip

rm *.zip

echo "*" > dataset/.gitignore