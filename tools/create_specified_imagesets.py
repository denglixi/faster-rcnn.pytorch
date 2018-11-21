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


def create_inner_imagesets():
    '''
    inner is the inner set between train of excl{dataset} and trainval of {dataset}
    '''
    cantten = ['Arts', 'Science', 'TechMixedVeg',
               'TechChicken', 'UTown', 'YIH']


    for ct in cantten:
        print("------processing {}-----------".format(ct))
        imgsets_path = "../data/Food/Food_{}/ImageSets".format(ct)
        anno_path = "../data/Food/Food_{}/Annotations".format(ct)
        for N in [0, 10, 30, 50, 100]:
            if N == 0:
                excl_class = get_categories("excl"+ct+"_train")
            else:
                excl_class = get_categories("excl"+ct+"_trainmt{}".format(N))
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
            print("saving inner mt {} sets:{}".format(N,len(filter_xmls)))
            print(imgsets_path)
            if N == 0:
                saving_file = "inner.txt"
            else:
                saving_file = "innermt{}.txt".format(N)
            with open(os.path.join(imgsets_path, saving_file), 'w') as f:
                for i in filter_xmls:
                    x_name = os.path.split(i)[1]
                    x_name = os.path.splitext(x_name)[0]
                    f.write(x_name + '\n')


def create_train_and_val_imagesets():
    """
    create train.txt and val.txt for the origin 6 canteens (Arts, science, techmix, techchicken , utown and yih)

    """
    cantten = ['Arts', 'Science', 'TechMixedVeg',
               'TechChicken', 'UTown', 'YIH']
    excl_canteen = ["excl"+x for x in cantten]
    cantten += excl_canteen
    cantten += ["All"]

    for ct in cantten:
        print("------processing {}-----------".format(ct))
        imgsets_path = "../data/Food/Food_{}/ImageSets".format(ct)
        anno_path = "../data/Food/Food_{}/Annotations".format(ct)

        with open(os.path.join(imgsets_path, "trainval.txt")) as f:
            trainval_content = f.readlines()
        train_content = []
        val_content = []

        for i, sample in enumerate(trainval_content):
            if i % 8 == 0 or i % 9 == 0:
                val_content.append(sample)
            else:
                train_content.append(sample)

        print("saving train sets:{}".format(len(train_content)))
        with open(os.path.join(imgsets_path, "train.txt"), 'w') as f:
            f.writelines(train_content)
        print("saving val sets:{}".format(len(val_content)))
        with open(os.path.join(imgsets_path, "val.txt"), 'w') as f:
            f.writelines(val_content)


def create_mtN_imagesets(N: int):
    """get_mtN_imagesets

    :param N: the least number of sample in each category
    :type N: int

    """
    cantten = ['Arts', 'Science', 'TechMixedVeg',
               'TechChicken', 'UTown', 'YIH']
    excl_canteen = ["excl"+x for x in cantten]
    cantten += excl_canteen
    cantten += ["All"]

    for ct in cantten:
        print("------processing {}-mt{}----------".format(ct, N))
        imgsets_path = "../data/Food/Food_{}/ImageSets".format(ct)
        anno_path = "../data/Food/Food_{}/Annotations".format(ct)
        imagesets = ['trainval', 'train', 'val']
       # imagesets = ['val']

        for imset in imagesets:
            with open(os.path.join(imgsets_path, imset+".txt"), 'r') as f:
                xml_files = [x.strip("\n")+'.xml' for x in f.readlines()]

            content = []
            for xf in xml_files:
                objects = parse_rec(os.path.join(anno_path, xf))
                for obj in objects:
                    if obj['name'] in get_categories(ct+"_{}_mt{}".format(imset, N)):
                        content.append(xf.split(".")[0] + '\n')
                        break

            print("saving {} sets:{}_mt{}".format(imset, len(content), N))
            with open(os.path.join(imgsets_path, "{}mt{}.txt".format(imset, N)), 'w') as f:
                f.writelines(content)
        #train_content = []
        #val_content = []

        # for i, sample in enumerate(trainval_content):
        #    if i % 8 == 0 or i % 9 == 0:
        #        val_content.append(sample)
        #    else:
        #        train_content.append(sample)

        #print("saving train sets:{}".format(len(train_content)))
        # with open(os.path.join(imgsets_path, "trainmt{}.txt".format(N)), 'w') as f:
        #    f.writelines(train_content)
        #print("saving val sets:{}".format(len(val_content)))
        # with open(os.path.join(imgsets_path, "valmt{}.txt".format(N)), 'w') as f:
        #    f.writelines(val_content)


def create_trainval_imagesets(path: str):
    """create_trainval_imagesets

    :param path: path of dataset
    :type path: str

    """
    anno_path = os.path.join(path, "Annotations")
    imsets_path = os.path.join(path, "ImageSets")
    xml_files = os.listdir(anno_path)
    with open(os.path.join(imsets_path, "trainval.txt"), 'w') as f:
        f.writelines([x.split(".")[0]+'\n' for x in xml_files])


if __name__ == '__main__':
    #create_train_and_val_imagesets()
    #for n in [10, 30, 50, 100]:
    #    create_mtN_imagesets(n)
    create_inner_imagesets()
