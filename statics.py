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


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def get_all_xml_files(root):
    allfiles = os.listdir(root)
    stats = {}
    for i in allfiles:
        objs = parse_rec(os.path.join(root, i))
        for obj in objs:
            cls_name = int(obj['name'])
            if cls_name in stats:
                stats[cls_name] += 1
            else:
                stats[cls_name] = 1

    stats = sorted(stats.items(), key=lambda t: t[0])
    return stats


def main():
    stats = get_all_xml_files("./data/Food/Food_YIH/Annotations")
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
