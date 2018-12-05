#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 denglixi <denglixi@xgpd1>
#
# Distributed under terms of the MIT license.

"""

"""

import xml.etree.ElementTree as ET
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from id2name import id2chn, id2eng
from xml_process import parse_rec

from food_category import get_categories


def get_all_xml_files_from_dir(dir_path):
    """get_all_xml_files_from_dir
    :param dir_path:
    """
    allfiles = os.listdir(dir_path)
    stats = {}
    for i in allfiles:
        objs = parse_rec(os.path.join(dir_path, i))
        for obj in objs:
            try:
                cls_name = int(obj['name'])
            except Exception as e:
                print(e)
                print(dir_path + i)

            if cls_name in stats:
                stats[cls_name] += 1
            else:
                stats[cls_name] = 1

    stats = sorted(stats.items(), key=lambda t: t[0])
    return stats


def get_all_xml_files_from_path(root_path):
    """get_all_xml_files_from_path
    recursive get all xmlfile from
    uparam root_path:
    """
    stats = {}
    for root, dirs, files in os.walk(root_path):
        for x_f in files:
            if os.path.splitext(x_f)[1] == '.xml':
                objs = parse_rec(os.path.join(root, x_f))
                for obj in objs:
                    try:
                        cls_name = int(obj['name'])
                    except Exception as e:
                        print(e)
                        print(root + x_f)

                    if cls_name in stats:
                        stats[cls_name] += 1
                    else:
                        stats[cls_name] = 1

    stats = sorted(stats.items(), key=lambda t: t[0])
    return stats


def get_xml_from_file(file_path, xml_dir):
    """get_xml_from_file

    :param file_path: train.txt
    :param xml_dir: Annotation dir
    """
    stats = {}
    with open(file_path) as f:
        all_xml_name = [x.strip()+".xml" for x in f.readlines()]
    try:
        for x_f in all_xml_name:
            objs = parse_rec(os.path.join(xml_dir, x_f))
            for obj in objs:
                cls_name = int(obj['name'])
                if cls_name in stats:
                    stats[cls_name] += 1
                else:
                    stats[cls_name] = 1
    except Exception:
        print("--------------")
        print(os.path.join(xml_dir, x_f))
        raise Exception
    stats = sorted(stats.items(), key=lambda t: t[0])
    return stats


def printdict(d):
    for k in d:
        print(k, end='    ')
    print("\n")
    for k in d:
        print(d[k], end='    ')
    print("\n")


def printlist_of_tuples(d):
    for k, v in d:
        print(k, end='    ')
    print("\n")
    for k, v in d:
        print(v, end='    ')
    print("\n")


def read_original_id(id_file_path):
    with open(id_file_path) as f:
        return [int(x.strip('\n')) for x in f.readlines()]


def is_id_in_annotation(xml_id, annotation_id):
    """is_id_in_annotation

    :param xml_id: list of id occured in xml files
    :param annotation_id: list of id which used for annotation
    """
    not_in_anno = [x for x in xml_id if x not in annotation_id]
    not_in_xml = [x for x in annotation_id if x not in xml_id]
    return not_in_anno, not_in_xml


def test_can(statisc, annotation_id):
    xml_id = [x[0] for x in statisc]
    return is_id_in_annotation(xml_id, annotation_id)


def statics_all_raw_data():

    # tech chicken
    techchicken_root = '/home/next/Big/data/Techno Edge/ChickenRice'
    techchicken_statis = get_all_xml_files_from_path(techchicken_root)
    print("-----------tech chicken----------")
    # printlist_of_tuples(techchicken_statis)
    print(test_can(techchicken_statis, read_original_id(
        './unified_id/techchicken_original_id.txt')))

    # Utown
    utown_root = '/home/next/Big/data/UTown_NoAC/MixedVeg/'
    utown_statis = get_all_xml_files_from_path(utown_root)
    print("-----------utown----------")
    # printlist_of_tuples(utown_statis)
    print(test_can(utown_statis, read_original_id(
        './unified_id/utown_original_id.txt')))

    # science
    science_root = '/home/next/Big/data/Science'
    science_statis = get_all_xml_files_from_path(science_root)
    print("-----------sciecce----------")
    # printlist_of_tuples(science_statis)
    #print( test_can(utown_statis, read_original_id('./unified_id/utown_original_id.txt')) )

    # arts
    arts_root = '/home/next/Big/data/Arts/'
    arts_statis = get_all_xml_files_from_path(arts_root)
    print("-----------arts----------")
    # printlist_of_tuples(arts_statis)
    print(sum([x[1] for x in arts_statis]))
    print(test_can(arts_statis, read_original_id(
        './unified_id/arts_original_id.txt')))

    # YIH
    xmlset = "/home/d/denglixi/data/Food/Food_YIH/ImageSets/occur_in_tech.txt"
    xml_dir = "/home/d/denglixi/data/Food/Food_YIH/Annotations"
    stats_yih = get_xml_from_file(xmlset, xml_dir)
    print("dict of YIH:")
    printlist_of_tuples(stats_yih)

    # tech statics
    tech_set = "/home/d/denglixi/data/Food/Food_Tech/ImageSets/val.txt"
    tech_xml_dir = "/home/d/denglixi/data/Food/Food_Tech/Annotations"
    stats_tech = get_xml_from_file(tech_set, tech_xml_dir)
    print("dict of tech:")
    printlist_of_tuples(stats_tech)


def get_statis(food_dataset_root, ct, img_set, cls_img_set):
    """get_statis

    :param food_dataset_root:
    :param ct: cantten name
    :param img_set: statistics the xml files whose name occur in img_set set
    :param cls_img_set: only statistic the categories occured in cls_img_set
    set
    """

    all_trainval_set = food_dataset_root + "Food_{}/ImageSets/{}.txt".format(
        ct, img_set)
    all_xml_dir = food_dataset_root + "Food_{}/Annotations".format(
        ct)
    all_stats = get_xml_from_file(all_trainval_set, all_xml_dir)
    all_stats = dict(all_stats)
    print("-------processing {} {}-----------".format(ct, img_set))
    imgset_category = get_categories(ct+'_'+img_set)
    with open("./statistics/{}_{}_{}_static.txt".format(ct, img_set, cls_img_set), 'w') as f:
        for cls in get_categories(cls_img_set)[1:]:
            if imgset_category is not None and cls not in imgset_category:
                continue
            if int(cls) in all_stats:
                k = int(cls)
                v = all_stats[k]
                f.write(str(k)+'\t'+str(v) + '\t' +
                        id2chn[str(k)] + '\t' + id2eng[str(k)] + '\n')
            else:
                f.write("\n")


def statistic_all():
    food_dataset_root = "/home/d/denglixi/faster-rcnn.pytorch/data/Food/"
    canttens = ['All', 'exclArts', 'exclYIH', 'exclTechChicken',
                'exclTechMixedVeg', 'exclUTown', 'exclScience',
                'YIH', 'Arts', 'Science', 'UTown',
                'TechChicken', 'TechMixedVeg']

    for ct in canttens:
        # construct imagesets
        imagesets = ['trainval', 'train', 'val', 'inner']
        imagesets_mt = []
        for N in [10, 30, 50, 100]:
            imagesets_mt += [x+"mt{}".format(N) for x in imagesets]
        imagesets += imagesets_mt

        # statistics
        for img_set in imagesets:
            if "inner" in img_set and ct not in ["Arts", "YIH", "UTown", "Science", "TechChicken", "TechMixedVeg"]:
                continue
            get_statis(food_dataset_root, ct, img_set)


def main():
    """main"""

    food_dataset_root = "/home/d/denglixi/faster-rcnn.pytorch/data/Food/"
    # for ct in ['YIH', 'exclYIH', 'All']:
    #    for sp in ['trainmt10', 'valmt10']:
    #        get_statis(food_dataset_root, ct, sp, ct+'_trainmt10')
    #get_statis(food_dataset_root, 'YIH', 'innerfew1mt10train', 'trainmt10')

    #get_statis(food_dataset_root, 'YIH', 'innerfew5mt10train', 'exclYIH_trainmt10')
    #get_statis(food_dataset_root, 'Arts', 'innerfew1mt10val', 'exclArts_trainmt10')
    get_statis(food_dataset_root, 'exclArts', 'trainmt10', 'Arts_inner')


if __name__ == '__main__':
    main()
