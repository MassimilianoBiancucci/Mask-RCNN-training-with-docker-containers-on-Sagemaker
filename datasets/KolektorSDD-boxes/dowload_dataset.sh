 
set -e

wget -O - "http://go.vicos.si/kolektorsddboxes" > archive.zip 

mkdir dataset

unzip -d dataset *.zip 

rm *.zip

echo "*" > dataset/.gitignore

