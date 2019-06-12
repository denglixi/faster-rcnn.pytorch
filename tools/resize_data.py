#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 dlx <dlx@dlx-next>
#
# Distributed under terms of the MIT license.

"""

"""
import os
import cv2
#from shutil import copyfile, copytree
from xml_process import parse_rec, read_xml, write_xml
import numpy as np
from distutils.dir_util import copy_tree
def resize_img_by_target_size(im, target_size, max_size):
    im = im.astype(np.float32, copy=False)
    # im = im[:, :, ::-1]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)

    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def resize_annotation(bbox, im_scale):
    return (bbox * im_scale).astype(np.int32)



Data_dir = '../data/Food/'
Save_dir = '../resize_data/Food'
canteens = ['Arts', 'Science', 'UTown', 'TechChicken', 'TechMixedVeg', 'YIH']

for ct in canteens:
    print('Processing {}'.format(ct))
    # set paths of dirs
    ct_jpeg_root = os.path.join(Data_dir, 'Food_{}/{}'.format(ct, 'JPEGImages'))
    ct_anno_root = os.path.join(Data_dir, 'Food_{}/{}'.format(ct, 'Annotations'))
    ct_imgset_root = os.path.join(Data_dir, 'Food_{}/{}'.format(ct, 'ImageSets'))
    save_ct_jpeg_root = os.path.join(Save_dir, 'Food_{}/{}'.format(ct, 'JPEGImages'))
    save_ct_anno_root = os.path.join(Save_dir, 'Food_{}/{}'.format(ct, 'Annotations'))
    save_ct_imgset_root = os.path.join(Save_dir, 'Food_{}/{}'.format(ct, 'ImageSets'))
    if not os.path.exists(save_ct_jpeg_root):
        os.makedirs(save_ct_jpeg_root)
    if not os.path.exists(save_ct_anno_root):
        os.makedirs(save_ct_anno_root)
    #if not os.path.exists(save_ct_imgset_root):
    #    os.makedirs(save_ct_imgset_root)

    # copy imageset
    copy_tree(ct_imgset_root, save_ct_imgset_root)

    # process each image and annotation
    count = 0
    for f in os.listdir(ct_jpeg_root):
        count += 1
        if count % 100 == 0:
            print("count:{}".format(count))
        f_name = os.path.splitext(f)[0]
        anno_f = f_name + '.xml'

        # file paths
        jpeg_path = os.path.join(ct_jpeg_root, f)
        anno_path = os.path.join(ct_anno_root, anno_f)
        save_jpeg_path = os.path.join(save_ct_jpeg_root, f)
        save_anno_path = os.path.join(save_ct_anno_root, anno_f)

        # resize image
        im = cv2.imread(jpeg_path)
        im , im_scale = resize_img_by_target_size(im, 600, 1000)
        cv2.imwrite(save_jpeg_path, im)

        #rewrite annotations
        if os.path.exists(anno_path):
            tree = read_xml(anno_path)
            for obj in tree.findall('object'):
                bbox = obj.find('bndbox')
                bbox_list = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
                bbox_list = np.array(bbox_list)
                rotated_bbox = resize_annotation(bbox_list, im_scale) 
                bbox.find('xmin').text = str(rotated_bbox[0])
                bbox.find('ymin').text = str(rotated_bbox[1])
                bbox.find('xmax').text = str(rotated_bbox[2])
                bbox.find('ymax').text = str(rotated_bbox[3]) 
            write_xml(tree, save_anno_path)



