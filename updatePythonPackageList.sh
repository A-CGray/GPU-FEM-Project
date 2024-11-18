#! /bin/bash

pip freeze > requirements.txt

# Comment out any lines that start with "-e git"
sed -i '/^-e git/s/^/#/' requirements.txt
