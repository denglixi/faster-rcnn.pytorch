#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 next <next@next-OptiPlex-990>
#
# Distributed under terms of the MIT license.

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib
import numpy as np
import math
import os
from tqdm import tqdm

from xml_process import parse_rec, read_xml, write_xml
'''
Since PIL do not rotate the image with oritation flag in exif infor, the annotation we labeled with labelimg was not in
the right position if we load the image with oritation flag (i.e. using cv2).
These function can convert the bounding boxes with oritation flag.
After rotating the bounding box, please use exiftran to rotate the coresponding images.
'''


def rotate_bbox(im_shape, bbox, exif_flag):
    """
    rotate one bbox base the exif flag and original image shape.
    :param im_shape: tuples, shape of origin image: (width, height)
    :param bbox:
    :param exif_flag: exif flag of orientation(274) :
    :return:
    """
    width, height = im_shape
    y = height
    x = width
    if exif_flag == 0 or exif_flag == 1:
        return bbox
    # 8 is clockwise 90 degree
    if exif_flag == 6:
        x1 = y - bbox[3]
        y1 = bbox[0]
        x2 = y - bbox[1]
        y2 = bbox[2]
    elif exif_flag == 8:
        x1 = bbox[1]
        y1 = x - bbox[2]
        x2 = bbox[3]
        y2 = x - bbox[0]
    elif exif_flag == 3:
        x1 = x - bbox[2]
        y1 = y - bbox[3]
        x2 = x - bbox[0]
        y2 = y - bbox[1]
    else:
        assert "not right flag"
    return [x1, y1, x2, y2]


def drawRectangles(xml, im):
    objects = parse_rec(xml)
    for o in objects:
        bbox = o['bbox']
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2],
                                               bbox[3]), color=(0, 255, 0), thickness=10)


def drawBoxWithPIL(im, xml):
    im = np.asarray(im)
    drawRectangles(xml, im)
    plt.imshow(im)
    plt.show()


def drawBoxWithCv2(im, xml):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    drawRectangles(xml, im)
    cv2.imshow("image", im)
    cv2.waitKey()


def rotate_bbox_in_xml(im_shape, exif_flag, xml_path):
    tree = read_xml(xml_path)
    for obj in tree.findall('object'):
        bbox = obj.find('bndbox')
        bbox_list = [int(bbox.find('xmin').text),
                     int(bbox.find('ymin').text),
                     int(bbox.find('xmax').text),
                     int(bbox.find('ymax').text)]
        rotated_bbox = rotate_bbox(im_shape, bbox_list, exif_flag)

        bbox.find('xmin').text = str(rotated_bbox[0])
        bbox.find('ymin').text = str(rotated_bbox[1])
        bbox.find('xmax').text = str(rotated_bbox[2])
        bbox.find('ymax').text = str(rotated_bbox[3])
    write_xml(tree, xml_path)


def test_rotate():
    # 1. read img and xml
    image_path = "./IMG_9613.JPG"
    xml_file = "./IMG_9613.xml"
    im = Image.open("./IMG_9613.JPG")
    # 2. rotate img and xml
    flag = 8  # set flag to 0,3,6,8 to test rotate
    print(im.size)
    if hasattr(im, '_getexif'):
        dict_exif = im._getexif()
        dict_exif[274] = flag
        if dict_exif[274] == 0:
            new_img = im
        if dict_exif[274] == 3:
            new_img = im.rotate(180, expand=True)
        if dict_exif[274] == 6:
            new_img = im.rotate(-90, expand=True)
        if dict_exif[274] == 8:
            new_img = im.rotate(90, expand=True)
    else:
        new_img = im
    print(new_img.size)
    # rotate bboxs
    rotate_bbox_in_xml(im.size, flag, xml_file)
    # 3. show rotate img and xml
    drawBoxWithPIL(new_img, "test.xml")
    exit()
    im = cv2.imread(image_path)
    drawBoxWithCv2(im, "test.xml")


def rotate_annotation_with_exif(root_dir):
    ori_flag_dict = {}
    ori_dir_dict = {6: {}, 8: {}, 1: {}, 0: {}, 3: {}}
    count = 0
    for _, _, fs in os.walk(root_dir):
        count += len(fs)
    pbar = tqdm(total=count)
    count = 0
    with open("oritation_images.log", 'w') as ori_img_log_f:
        for root, _, files in os.walk(root_dir):
            for file_i in files:
                count += 1
                if count % 100 == 0:
                    pbar.update(100)
                f_path = os.path.join(root, file_i)
                file_name, ex_name = os.path.splitext(f_path)
                if ex_name == '.JPG' or ex_name == '.jpg':
                    xml_path = os.path.join(root, file_name + '.xml')
                    if os.path.exists(xml_path):
                        # if a image file have a annotation file ,then the image will be processed.
                        # 1. read image, get shape and exif_flag
                        # 2. rotate bbox and write xml
                        im = Image.open(f_path)
                        try:
                            if hasattr(im, '_getexif'):
                                dict_exif = im._getexif()
                                orientation_flag = dict_exif[274]
                                if orientation_flag not in [0, 1]:
                                    ori_img_log_f.write(
                                        str(orientation_flag)+'\t' + f_path + '\n')
                                #rotate_bbox_in_xml(im.size, orientation_flag,xml_path)
                                # statistics
                                if orientation_flag not in ori_flag_dict:
                                    ori_flag_dict[orientation_flag] = 1
                                else:
                                    ori_flag_dict[orientation_flag] += 1
                                if root not in ori_dir_dict[orientation_flag]:
                                    ori_dir_dict[orientation_flag][root] = 1
                        except (KeyError, TypeError):
                            with open("no_oritation_flag.log", 'a') as f:
                                f.write(f_path+'\n')

                            # break
                        else:
                            continue
                    else:
                        continue
    pbar.close()
    print(ori_dir_dict)
    print(ori_flag_dict)


def test_oritation_of_dataset():
    root_dir = "/home/next/Big/SG"
    root_dir = "/home/next/Big/data"
    # rotate_annotation_with_exif(root_dir)
    test_dir = "/home/next/Big/test_oritation/11oct_DONE328"
    for f in os.listdir(test_dir):
        f = os.path.join(test_dir, f)
        file_name, ex_name = os.path.splitext(f)
        if ex_name == '.JPG' or ex_name == '.jpg':
            xml_path = os.path.join(test_dir, file_name + '.xml')
            if os.path.exists(xml_path):
                # im = cv2.imread(f)
                # drawBoxWithCv2(im , xml_path)
                im = Image.open(f)
                drawBoxWithPIL(im, xml_path)
                import pdb
                pdb.set_trace()
                if hasattr(im, '_getexif'):
                    dict_exif = im._getexif()
                    orientation_flag = dict_exif[274]


if __name__ == '__main__':
    root_dir = "/home/next/Big/data/"
    # rotate_annotation_with_exif(root_dir)

    img = '/home/next/Big/data/YIH/17oct_YUNAN_DONE140/IMG_20181017_121015.jpg'
    xml = '/home/next/Big/data/YIH/17oct_YUNAN_DONE140/IMG_20181017_121015.xml'

    # drawBoxWithPIL(Image.open(img), xml)
    # drawBoxWithCv2(cv2.imread(img), xml)
