#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 denglixi <denglixi@xgpd1>
#
# Distributed under terms of the MIT license.

"""
transform the splited anonotations of each carteen to the uniform labels
"""

import os
import argparse
from xml_process import read_xml, write_xml


def construct_dict(key_file, val_file):
    res = {}
    with open(key_file, 'r') as f:
        keys = [x.strip('\n') for x in f.readlines()]
    with open(val_file, 'r') as f:
        vals = [x.strip('\n') for x in f.readlines()]
    for k, v in zip(keys, vals):
        res[k] = v
    return res


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_path", type=str,
                        default="./", dest="process_path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()

    path = "/home/d/denglixi/data/Food/YIH/20sept"
    for root, dirs, files in os.walk(path):
        for x_f in files:
            if os.path.splitext(x_f)[1] == '.xml':
                print(root+x_f)
                tree = read_xml(os.path.join(root, x_f))
                objects = tree.findall('object')
                convert_dict = construct_dict('./key.txt', './value.txt')
                for obj in objects:
                    obj.find('name').text = convert_dict[obj.find('name').text]

                write_xml(tree, os.path.join(root, x_f))
