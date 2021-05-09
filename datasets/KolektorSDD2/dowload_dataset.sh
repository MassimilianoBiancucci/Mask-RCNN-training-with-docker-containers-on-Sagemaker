 
set -e

wget -O - "http://go.vicos.si/kolektorsdd2" > archive.zip 

mkdir dataset

unzip -d dataset/ *.zip 

rm *.zip

echo "*" > dataset/.gitignore
