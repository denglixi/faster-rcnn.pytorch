#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 denglixi <denglixi@xgpd7>
#
# Distributed under terms of the MIT license.

"""

"""
from xml_process import parse_rec
import numpy as np
import os
import pdb

canteens = ['All']

coocurence_matrix = np.zeros((21,21))

index_count = 0
cls_to_index = {}

#anno_dir = "../data/Food/Food_All/Annotations/"
anno_dir = "../data/school_lunch_dataset/Annotations/"
anno_list = os.listdir(anno_dir)
for anno in anno_list:
    objs = parse_rec(os.path.join(anno_dir, anno))
    all_cls = []
    for obj in objs:
        obj_name = obj['name']
        if obj_name not in cls_to_index:
            cls_to_index[obj_name] = index_count
            index_count += 1
        all_cls.append(obj['name'])
    all_cls = set(all_cls)

    for cls_i in all_cls:
        for cls_j in all_cls:
            coocurence_matrix[cls_to_index[cls_i]][cls_to_index[cls_j]] += 1

print(cls_to_index.keys())
with open("matrix_name_sl.txt",'w') as f:
    f.writelines([i+'\n' for i in cls_to_index])
with open("coocurence_matrix_sl.txt" , 'w') as f:
    for i in range(len(coocurence_matrix)):
        for j in range(len(coocurence_matrix[i])):
            f.write(str(int(coocurence_matrix[i][j])) + '\t')
        f.write('\n')

print("done")


def get_cooc_count(cls_name_i, cls_name_j):
    index_i = cls_to_index[cls_name_i]
    index_j = cls_to_index[cls_name_j]
    cooc_count = coocurence_matrix[index_i][index_j]
    return cooc_count








