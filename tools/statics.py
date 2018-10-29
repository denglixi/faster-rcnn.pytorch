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
from xml_process import parse_rec


def get_all_xml_files_from_dir(dir_path):
    """get_all_xml_files_from_dir
    获取某个文件夹下所有xml文件

    :param dir_path:
    """
    allfiles = os.listdir(dir_path)
    stats = {}
    for i in allfiles:
        objs = parse_rec(os.path.join(dir_path, i))
        for obj in objs:
            cls_name = int(obj['name'])
            if cls_name in stats:
                stats[cls_name] += 1
            else:
                stats[cls_name] = 1

    stats = sorted(stats.items(), key=lambda t: t[0])
    return stats


def get_all_xml_files_from_path(root_path):
    """get_all_xml_files_from_path
    获取某个路径下，所有循环文件夹的xml文件

    :param root_path:
    """
    stats = {}
    for root, dirs, files in os.walk(root_path):
        for x_f in files:
            if os.path.splitext(x_f)[1] == '.xml':
                objs = parse_rec(os.path.join(root, x_f))
                for obj in objs:
                    cls_name = int(obj['name'])
                    if cls_name in stats:
                        stats[cls_name] += 1
                    else:
                        stats[cls_name] = 1

    stats = sorted(stats.items(), key=lambda t: t[0])
    return stats


def get_xml_from_file(file_path, xml_dir):
    """get_xml_from_file
    根据文件内容，获取文件内容中给所有xml文件

    :param file_path: 文件路径，每一行一个xml文件名，不带后缀
    :param xml_dir:xml所在的文件夹
    """
    stats = {}
    with open(file_path) as f:
        all_xml_name = [x.strip()+".xml" for x in f.readlines()]
    for x_f in all_xml_name:
        objs = parse_rec(os.path.join(xml_dir, x_f))
        for obj in objs:
            cls_name = int(obj['name'])
            if cls_name in stats:
                stats[cls_name] += 1
            else:
                stats[cls_name] = 1
    stats = sorted(stats.items(), key=lambda t: t[0])
    return stats


def main():
    # stats = get_all_xml_files_from_path(
    #    "/home/d/denglixi/data/Food/YIH/19sept")
    # stats = get_all_xml_files_from_path(
    #    "/home/d/denglixi/data/Food/YIH/20sept")
    xmlset = "/home/d/denglixi/data/Food/Food_Tech/ImageSets/train.txt"
    xml_dir = "/home/d/denglixi/data/Food/Food_Tech/Annotations"
    stats = get_xml_from_file(xmlset, xml_dir)

    zhfont1 = FontProperties(fname='./simsun.ttc')
    for v, k in stats:
        print(v, end='\t')
    print("\n")
    for v, k in stats:
        print(k, end='\t')

    return
    #plt.rcParams['font.sans-serif'] = ['simsum.ttc']
    plt.bar(list(range(len(stats))), [v for k, v in stats])
    plt.xticks(list(range(len(stats))), [u"白饭 ", u"咸蛋 ", u"翻炒蛋 ", u"番茄蛋 ",
                                         u"糙米 ", "瘦肉 ", "小白菜（绿） ", "包菜 ", "苦瓜 ", "芹菜 ", "番薯葉 ",
                                         u"南瓜 ", "豆芽 ", "白萝卜 ", "长豆 ", "西兰花和菜花 ", "酸辣羊角豆/秋葵 ", "菜花 ",
                                         u"炒豆腐 ", "猪肝 ", "土豆炒番茄酱豆 ", "茄子 ", "滷豆腐 ", "鸭肉 ", "咖喱鸡 ",
                                         u"红酱。酸甜肉 ", "肉碎 ", "卤鸡 ", "猪脚 ", "排骨 ", "三层肉 ", "黑酱猪肉",
                                         u"炸鱼。整只 ", "炸鱼片。红酱汁。看似炒蛋 ", "章鱼，鱿鱼。O型 ", "糖醋鱼", "炸鱼片，粉红酱汁 ",  "毛瓜丝"], fontproperties=zhfont1, rotation="vertical")
    plt.show()
    print(stats)
    pass


if __name__ == '__main__':
    main()
