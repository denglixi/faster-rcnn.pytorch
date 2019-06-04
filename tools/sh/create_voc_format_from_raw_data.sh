#! /bin/sh
#
# createdataset.sh
# Copyright (C) 2018 denglixi <denglixi@xgpd1>
#
# Distributed under terms of the MIT license.
#
# shell to create voc dataset from rawdata we collect
# 1. convert .JPG to .jpg
# 2. create VOC format dataset

RAW_DATA_PATH="/home/d/denglixi/rawdata/EconomicBeeHoon"
DATA_PATH="/home/d/denglixi/faster-rcnn.pytorch/data/EconomicBeeHoon"

echo $RAW_DATA_PATH
mkdir -p $DATA_PATH

find "$RAW_DATA_PATH" -name "*.JPG" | awk -F "." '{print $1}' | xargs -i -t mv {}.JPG {}.jpg

echo  'count of raw annotation'
find "$RAW_DATA_PATH" -name "*.xml" | wc -l
echo  'count of raw images'
find "$RAW_DATA_PATH" -name "*.jpg" | wc -l


python ./create_voc_format_from_raw_data.py --raw_path "$RAW_DATA_PATH" --save_path "$DATA_PATH"

echo  'count of processed annotation'
find "$DATA_PATH" -name "*.xml" | wc -l
echo  'count of processed images'
find "$DATA_PATH" -name "*.jpg" | wc -l

#find $RAW_DATA_PATH -name "*.xml" -exec cp {} ./Annotations \;
#find ./ -name "*.jpg" -exec cp {} ./JPEGImages \;

# find ./ -name "./Annotations/*.xml" | awk -F "." '{print $2}' >>trainval.txt
