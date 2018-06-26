#!/bin/zsh
#!/home/jinseok/anaconda3/bin/python3

chmod +x remove_x00.sh

python3 remove_x00.py

find . -type f -name '*[0-9].txt' -delete
for file in *_new.txt
do
  mv -i ${file} ${file/_new/}
done
