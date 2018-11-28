#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 denglixi <denglixi@xgpd8>
#
# Distributed under terms of the MIT license.

"""

"""

import xml.etree.ElementTree as ET
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from statics import get_xml_from_file


def create_category(imageset):
    canttens = ['All', 'exclArts', 'exclYIH', 'exclTechChicken',
                'exclTechMixedVeg', 'exclUTown', 'exclScience',
                'YIH', 'Arts', 'Science', 'UTown',
                'TechChicken', 'TechMixedVeg']
    category_file = './food_category.py'
    food_dataset_root = "/home/d/denglixi/faster-rcnn.pytorch/data/Food/"
    datasets = imageset
    with open(category_file, 'w') as f:
        f.write("def get_categories(category):\n")
        for ct in canttens:
            for dataset in datasets:
                if dataset == "inner" and ct not in ["Arts", "YIH", "UTown", "Science", "TechChicken", "TechMixedVeg"]:
                    continue
                all_trainval_set = food_dataset_root + "Food_{}/ImageSets/{}.txt".format(
                    ct, dataset)
                all_xml_dir = food_dataset_root + "Food_{}/Annotations".format(
                    ct)
                all_stats = get_xml_from_file(all_trainval_set, all_xml_dir)
                # printlist_of_tuples(all_stats)

                f.write('    if category == \'{}_{}\':\n        return [\'__background__\', '.format(
                    ct, dataset), )
                for k, v in all_stats:
                    f.write("'{}', ".format(k))
                f.write("]\n")

                N = [10, 30, 50, 100]
                for n in N:
                    # only add the category whose number is more than n
                    f.write('    if category == \'{}_{}_mt{}\':\n        return [\'__background__\', '.format(
                        ct, dataset, n), )
                    for k, v in all_stats:
                        if v > n:
                            f.write("'{}', ".format(k))
                    f.write("]\n")

                N = [10, 30, 50, 100]
                for n in N:
                    # only add the category whose number is more than n
                    f.write('    if category == \'{}_{}mt{}\':\n        return [\'__background__\', '.format(
                        ct, dataset, n), )
                    for k, v in all_stats:
                        if v > n:
                            f.write("'{}', ".format(k))
                    f.write("]\n")
    return


def main():
    #create_category(["trainval", "train", "val"])
    # create_category(["inner"])
    create_category(["trainval", "train", "val", "inner"])


if __name__ == '__main__':
    main()
