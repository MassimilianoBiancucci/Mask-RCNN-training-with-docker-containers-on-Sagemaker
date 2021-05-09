 
set -e

kaggle datasets download -d mhskjelvareid/dagm-2007-competition-dataset-optical-inspection

mkdir dataset

unzip -d dataset/ *.zip 

rm *.zip

echo "*" > dataset/.gitignore
