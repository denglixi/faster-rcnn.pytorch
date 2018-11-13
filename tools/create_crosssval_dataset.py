#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 lixi <lixi@LMS-SGPU>
#
# Distributed under terms of the MIT license.

"""

"""
import os
food_dataset_dir = "/home/d/denglixi/faster-rcnn.pytorch/data/Food/"

canteens = ['Arts', 'Science', 'TechChicken', 'TechMixedVeg', 'UTown', 'YIH']


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


for exclude_ct in canteens + ['All']:
    print("canteens: exclude ", exclude_ct)
    # create
    if exclude_ct == 'All':
        exclude_food_root = os.path.join(food_dataset_dir, 'Food_'+exclude_ct)
    else:
        exclude_food_root = os.path.join(
            food_dataset_dir, 'Food_excl'+exclude_ct)
    exclude_food_Anno_dir = os.path.join(exclude_food_root, "Annotations")
    exclude_food_ImSet_dir = os.path.join(exclude_food_root, "ImageSets")
    exclude_food_JPEG_dir = os.path.join(exclude_food_root, "JPEGImages")
    create_dir(exclude_food_Anno_dir)
    create_dir(exclude_food_JPEG_dir)
    create_dir(exclude_food_ImSet_dir)

    exclude_trainval_path = os.path.join(
        exclude_food_ImSet_dir, 'trainval.txt')
    trainval_content = []

    for ct in canteens:
        if exclude_ct == ct:
            continue
        ct_root = os.path.join(food_dataset_dir, 'Food_' + ct)
        ct_Anno_dir = os.path.join(ct_root, 'Annotations')
        ct_ImSet_dir = os.path.join(ct_root, 'ImageSets')
        ct_JPEG_dir = os.path.join(ct_root, 'JPEGImages')
        # 处理空格
        # create soft link for mixed datset
        for f in os.listdir(ct_Anno_dir):
            os.symlink(ct_Anno_dir+'/' + f, exclude_food_Anno_dir + '/' + f)
        for f in os.listdir(ct_JPEG_dir):
            os.symlink(ct_JPEG_dir+'/' + f, exclude_food_JPEG_dir+'/' + f)
        # trainval.txt
        ct_trainval_path = os.path.join(ct_ImSet_dir, 'trainval.txt')
        with open(ct_trainval_path) as f:
            trainval_content += f.readlines()
    print(len(trainval_content))
    with open(exclude_trainval_path, 'w') as f:
        f.writelines(trainval_content)

    train_content = []
    val_content = []
    # TODO: the images of one same dish which were taken from different angles should be splited.
    for i, sample in enumerate(trainval_content):
        if i % 8 == 0 or i % 9 == 0:
            val_content.append(sample)
        else:
            train_content.append(sample)

    with open(os.path.join(exclude_food_ImSet_dir, 'train.txt'), 'w') as f:
        print("len of training set", len(train_content))
        f.writelines(train_content)
    with open(os.path.join(exclude_food_ImSet_dir, 'val.txt'), 'w') as f:
        print("len of val set", len(val_content))
        f.writelines(val_content)


def split_train_val(imgsets_path):
    trainval_file = os.path.join(imgsets_path, "trainval.txt")
    train_file = os.path.join(imgsets_path, "train.txt")
    val_file = os.path.join(imgsets_path, "val.txt")
    trainval_content = []
    with open(trainval_file) as f:
        pass
