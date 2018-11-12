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
from xml_process import parse_rec

def travel_xml_in_path(callback_func):
    """
    travel
    callback_func : accept a function with xml object
    """
    for root, dirs, files in os.walk(root_path):
        for x_f in files:
            if os.path.splitext(x_f)[1] == '.xml':
                callback_func()



def get_all_xml_files_from_path(root_path):
    """get_all_xml_files_from_path
    recursive get all xmlfile from
    uparam root_path:
    """
    stats = {}
    log_file = open("check_box.log", 'w')
    for root, dirs, files in os.walk(root_path):
        for x_f in files:
            if os.path.splitext(x_f)[1] == '.xml':
                objs = parse_rec(os.path.join(root, x_f))
                for obj in objs:
                    bbox = obj['bbox']
                    if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == 0 or bbox[3] ==0:
                        log_file.write(os.path.join(root, x_f) + '\t' + "0" + '\n')
                        log_file.write(str(bbox) + '\n')
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        log_file.write(os.path.join(root, x_f) + '\t' + "1" + '\n')
                        log_file.write(str(bbox) + '\n')

get_all_xml_files_from_path("/home/lixi/data/FoodDataset/Food_All/Annotations")
