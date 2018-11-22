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


def main():
    """main"""

    #canttens = ['All', 'exclArts', 'exclYIH', 'exclTechChicken',
    #            'exclTechMixedVeg', 'exclUTown', 'exclScience',
    #            'YIH', 'Arts', 'Science', 'UTown',
    #            'TechChicken', 'TechMixedVeg']

    canttens = ['All', 'exclYIH',
                'exclUTown',
                'YIH', 'UTown',
                ]
    food_dataset_root = "/home/d/denglixi/faster-rcnn.pytorch/data/Food/"
    for ct in canttens:
        datasets = ['trainval', 'train', 'val', 'inner']
        dataset_mt = []
        for N in [10, 30, 50, 100]:
            dataset_mt += [x+"mt{}".format(N) for x in datasets]
        datasets += dataset_mt
        #datasets = ['trainval']
        for dataset in datasets:
            if "inner" in dataset and ct not in ["Arts", "YIH", "UTown", "Science", "TechChicken", "TechMixedVeg"]:
                continue

            all_trainval_set = food_dataset_root + "Food_{}/ImageSets/{}.txt".format(
                ct, dataset)
            all_xml_dir = food_dataset_root + "Food_{}/Annotations".format(
                ct)
            all_stats = get_xml_from_file(all_trainval_set, all_xml_dir)
            # printlist_of_tuples(all_stats)
            print("-------processing {} {}-----------".format(ct, dataset))

            with open("./statistics/{}_{}_static.txt".format(ct, dataset), 'w') as f:
                for k, v in all_stats:
                    if str(k) in get_categories(ct+'_'+dataset):
                        f.write(str(k)+'\t'+str(v) + '\t' +
                                id2chn[str(k)] + '\t' + id2eng[str(k)] + '\n')

    return
    #chn_font = FontProperties(fname='./simsun.ttc')
    #plt.bar(list(range(len(stats))), [v for k, v in stats])
    #plt.xticks(list(range(len(stats))), [u"白饭 ", u"咸蛋 ", u"翻炒蛋 ", u"番茄蛋 ",
    #                                     u"炸鱼。整只 ", "炸鱼片。红酱汁。看似炒蛋 ", "章鱼，鱿鱼。O型 ", "糖醋鱼", "炸鱼片，粉红酱汁 ",  "毛瓜丝"], fontproperties=chn_font, rotation="vertical")
    #plt.show()
    #print(stats)
    pass


if __name__ == '__main__':
    main()
