 
set -e

kaggle datasets download -d mhskjelvareid/dagm-2007-competition-dataset-optical-inspection

mv dagm-2007-competition-dataset-optical-inspection dataset

unzip *.zip 

rm *.zip

echo "*" > dataset/.gitignore
