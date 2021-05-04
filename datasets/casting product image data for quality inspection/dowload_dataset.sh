 
set -e

kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product

unzip *.zip 

rm *.zip

for d in */ ; do
    echo "*" > $d/.gitignore
done
