#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 denglixi <denglixi@xgpd0>
#
# Distributed under terms of the MIT license.

"""
获取特定的imageset 文件
1. 读xml
2. 过滤
    实现了按照obejct的类别过滤
"""
import os
from xml_process import parse_rec

from food_category import get_categories


def process_all_xml_files_from_dir(dir_path, call_back_func):
    """process_all_xml_files_from_dir

    :param dir_path: path of xml
    :param call_back_func: call back function to process the dir
                            function accpet three params:
                                xml_path, objs, **kwargs
    """
    allfiles = os.listdir(dir_path)
    stats = {}
    for i in allfiles:
        xml_path = os.path.join(dir_path, i)
        objs = parse_rec(xml_path)
        call_back_func(xml_path, objs)


class filter_xml:
    def __init__(self, cls_dict):
        self.cls_dict = cls_dict
        self.reserver_xmls = []

    def process(self, xml_path, objs):
        for obj in objs:
            if obj['name'] in self.cls_dict:
                self.reserver_xmls.append(xml_path)
                break

    def clean(self):
        self.cls_dict = []


def clo(reserver_class):
    """clo:这是一个闭包, nonlocal 的列表保留 过滤后的 值"""

    filter_xml = []

    def filter_xml1(xml_path, objs):
        """filter_xml

        :param xml_path:
        :param objs:
        :param **kwargs:
        """
        # filtered_xml = kwargs['filter_xml_list']
        # reserver_class = kwargs['reserve_cls_dict']
        nonlocal filter_xml
        for obj in objs:
            if obj['name'] in reserver_class:
                filter_xml.append(xml_path)
                break
    return filter_xml1


def main():
    cantten = ['Arts', 'Science', 'TechMixedVeg',
               'TechChicken', 'UTown', 'YIH']

    for ct in cantten:
        imgsets_path = "../data/Food/Food_{}/ImageSets".format(ct)
        anno_path = "../data/Food/Food_{}/Annotations".format(ct)
        excl_class = get_categories("excl"+ct+"_train")
        # 3种方法实现通过回调函数，对xml进行筛选
        # 1. save extra info of callback with class
        # fx = filter_xml(tech_classes)
        # process_all_xml_files_from_dir(path, fx.process)
        # print(len(fx.reserver_xmls))

        # 2. save extra info of callback with closet
        fx_clo = clo(excl_class)
        process_all_xml_files_from_dir(anno_path, fx_clo)
        # print(len(fx_clo.__closure__))  # __closure__ 有cell对象的元祖构成
        filter_xmls = fx_clo.__closure__[
            0].cell_contents  # cell 对象有cell_contents的内容

        # 3. 通过协程
        # how to implement??
        # NotImplemented

        # 保存筛选信息
        with open(os.path.join(imgsets_path, "inner.txt"), 'w') as f:
            for i in filter_xmls:
                x_name = os.path.split(i)[1]
                x_name = os.path.splitext(x_name)[0]
                f.write(x_name + '\n')


if __name__ == '__main__':
    main()
