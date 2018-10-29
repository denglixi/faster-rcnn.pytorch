#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 denglixi <denglixi@xgpd1>
#
# Distributed under terms of the MIT license.

"""
从收集的数据构建voc格式的数据集
1. 按照
    DATASET
        Annotations
        ImageSets
        JPEGImages
的格式重新安排数据
2. 对于重复文件（不同日期收集可能图片名称重复）重命名
"""
import os
import shutil
import argparse


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", dest="raw_path", type=str)
    parser.add_argument("--save_path", dest="save_path",
                        type=str)
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    #RAW_DATA_PATH = './YIH'
    #DATA_PATH = "./Food_YIH/"
    RAW_DATA_PATH = args.raw_path
    DATA_PATH = args.save_path
    anno_path = os.path.join(DATA_PATH, 'Annotations')
    jpeg_path = os.path.join(DATA_PATH, 'JPEGImages')

    if not os.path.exists(anno_path):
        os.makedirs(anno_path)
    if not os.path.exists(jpeg_path):
        os.makedirs(jpeg_path)

    # rename jpg and xml
    # save to the voc dirs
    for root, _, filenames in os.walk(RAW_DATA_PATH):
        for f_name in filenames:
            file_path = os.path.join(root, f_name)
            ext = os.path.splitext(file_path)[1]
            split_file_path = file_path.split("/")

            if ext == '.jpg':
                jpeg_save_path = os.path.join(
                    jpeg_path, split_file_path[-2] + f_name)
                shutil.copyfile(file_path, jpeg_save_path)

            elif ext == '.xml':
                xml_save_path = os.path.join(
                    anno_path, split_file_path[-2] + f_name)
                shutil.copyfile(file_path, xml_save_path)

    # save all xml name to ImageSets dir
    imageset_path = os.path.join(DATA_PATH, 'ImageSets')
    if not os.path.exists(imageset_path):
        os.makedirs(imageset_path)
    with open(os.path.join(imageset_path, "allxml.txt"), "w") as f:
        for f_name in os.listdir(anno_path):
            split_f_name = os.path.splitext(f_name)
            ext = split_f_name[1]
            if ext == '.xml':
                f.write(split_f_name[0] + '\n')


if __name__ == '__main__':
    main()
    pass
