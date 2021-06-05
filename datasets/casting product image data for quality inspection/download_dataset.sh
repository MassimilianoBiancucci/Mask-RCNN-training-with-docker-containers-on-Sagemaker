 
set -e

kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product

mkdir dataset

unzip -d dataset/ *.zip 

rm *.zip

echo "*" > dataset/.gitignore
