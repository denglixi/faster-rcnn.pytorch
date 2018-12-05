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
from clean_val import create_dishes


def process_all_xml_files_from_dir(dir_path, call_back_func):
    """process_all_xml_files_from_dir

    :param dir_path: path of xml
    :param call_back_func: call back function to process the dir
                            function accpet three params:
                                xml_path, objs, **kwargs
    """
    allfiles = sorted(os.listdir(dir_path))
    for i in allfiles:
        xml_path = os.path.join(dir_path, i)
        objs = parse_rec(xml_path)
        call_back_func(xml_path, objs)


def process_xml_from_file(file_path, xml_dir, call_back_func):
    """get_xml_from_file

    :param file_path: train.txt
    :param xml_dir: Annotation dir
    """
    with open(file_path) as f:
        all_xml_name = [x.strip()+".xml" for x in f.readlines()]

    for x_f in all_xml_name:
        xml_path = os.path.join(xml_dir, x_f)
        objs = parse_rec(xml_path)
        call_back_func(xml_path, objs)


class Xml_in_few_sample_filter:
    """Xml_in_few_sample_filter
    class for find few_count samples whose categories are match cls_dict
    """

    def __init__(self, cls_dict, few_count):
        self.cls_dict = cls_dict
        self.reserver_xmls = []
        self.discard_xmls = []
        self.few_count = few_count

    def process(self, xml_path, objs):
        x_name = os.path.split(xml_path)[1]
        x_name = os.path.splitext(x_name)[0]
        for obj in objs:
            if obj['name'] in self.cls_dict and self.cls_dict[obj['name']] < self.few_count:
                self.reserver_xmls.append(x_name)
                for o in objs:
                    try:
                        self.cls_dict[o['name']] += 1
                    except Exception:
                        print("{} not be found".format(o['name']))
                return
        self.discard_xmls.append(x_name)

    def clean_discard_by_dishes(self, dishes):
        """clean_discard_by_dishes

        clean image names
        :param dishes: list of dish which is list of image names
        """
        discard_xmls = []
        for img_name in self.discard_xmls:
            for dish in dishes:
                if img_name in dish:
                    others_in_reserved = False
                    for img in dish:
                        if img in self.reserver_xmls:
                            others_in_reserved = True
                            break
                    if not others_in_reserved:
                        discard_xmls.append(img_name)
                        break
        self.discard_xmls = discard_xmls


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


def filter_clo(reserver_class):
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


def create_inner_imageset(ct, N):
    '''
    inner is the inner set between train of excl{dataset} and trainval of {dataset}
    '''
    print("------processing {}-----------".format(ct))
    imgsets_path = "../data/Food/Food_{}/ImageSets".format(ct)
    anno_path = "../data/Food/Food_{}/Annotations".format(ct)

    if N == 0:
        excl_class = get_categories("excl"+ct+"_train")
    else:
        excl_class = get_categories("excl"+ct+"_trainmt{}".format(N))
    # 3种方法实现通过回调函数，对xml进行筛选
    # 1. save extra info of callback with class

    fx = filter_xml(excl_class)
    process_all_xml_files_from_dir(anno_path, fx.process)
    print(len(fx.reserver_xmls))
    filter_xmls = fx.reserver_xmls

    # 保存筛选信息
    print("saving inner mt {} sets:{}".format(N, len(filter_xmls)))
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


def create_few_inner_for_train_val(ct, imgset, mtN, fewN):
    """select_few_inner_for_train

    :param ct:
    :param mtN: N of mt which means the number of training sample is more than N
    :param fewN: the number of selected sample for each categories
    """
    print("------processing {}-selecting few inner--------".format(ct))
    imgsets_path = "../data/Food/Food_{}/ImageSets".format(ct)
    anno_path = "../data/Food/Food_{}/Annotations".format(ct)
    imset_path = os.path.join(imgsets_path, imgset+'.txt')

    if mtN == 0:
        excl_classes = get_categories("excl"+ct+"_train")
    else:
        excl_classes = get_categories("excl"+ct+"_trainmt{}".format(mtN))

    cls_sample_count = {}
    for ex_cls in excl_classes[1:]:
        cls_sample_count[ex_cls] = 0

    few_filter = Xml_in_few_sample_filter(cls_sample_count, fewN)
    dishes = create_dishes(ct)
    process_xml_from_file(imset_path, anno_path,
                          few_filter.process)

    # 保存筛选信息

    def saving_file(xmls, imgset):
        print("saving inner few{} mt {} {} sets:{}".format(
            fewN, mtN, imgset, len(xmls)))

        if mtN == 0:
            saving_file = "innerfew{}{}.txt".format(fewN, imgset)
        else:
            saving_file = "innerfew{}mt{}{}.txt".format(fewN, mtN, imgset)
        with open(os.path.join(imgsets_path, saving_file), 'w') as f:
            for x_name in xmls:
                f.write(x_name + '\n')

    few_filter.clean_discard_by_dishes(dishes)
    saving_file(few_filter.reserver_xmls, 'train')
    saving_file(few_filter.discard_xmls, 'val')


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
            fx_clo = filter_clo(excl_class)
            process_all_xml_files_from_dir(anno_path, fx_clo)
            # print(len(fx_clo.__closure__))  # __closure__ 有cell对象的元祖构成
            filter_xmls = fx_clo.__closure__[
                0].cell_contents  # cell 对象有cell_contents的内容

            # 3. 通过协程
            # how to implement??
            # NotImplemented

            # 保存筛选信息
            print("saving inner mt {} sets:{}".format(N, len(filter_xmls)))
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


def create_train_and_val_imagesets(canteen, train_content, val_content):
    """create_train_and_val_imagesets
    create train.txt and val.txt for canteen
    :param canteen:
    :param train_content:
    :param val_content:
    """

    print("------processing {}-----------".format(canteen))
    root_path = '../data/Food/Food_{}/'.format(canteen)
    imgsets_path = os.path.join(root_path, 'ImageSets')

    train_content = [x+'\n' for x in train_content]
    val_content = [x+'\n' for x in val_content]
    print("saving train sets:{}".format(len(train_content)))
    with open(os.path.join(imgsets_path, "train.txt"), 'w') as f:
        f.writelines(train_content)

    print("saving val sets:{}".format(len(val_content)))
    with open(os.path.join(imgsets_path, "val.txt"), 'w') as f:
        f.writelines(val_content)


def split_dishes2train_val(dishes):
    """splist_dishes2train_val

    :param dishes:
    :return (train_content: list, val_content: list)
    """
    train_content = []
    val_content = []
    for i, dish in enumerate(dishes):
        if i % 5:
            train_content += dish
        else:
            val_content += dish
    return train_content, val_content


def create_mtN_imageset(canteen, imgset, N: int):
    """create_mtN_imageset

    :param canteen:
    :param imgset: only support train or val
    :param N:
    :type N: int
    """

    assert imgset != 'train' or imgset != 'val'
    print("---processing {} mt {} {} ------".format(canteen, imgset, N))

    imgsets_path = "../data/Food/Food_{}/ImageSets".format(canteen)
    anno_path = "../data/Food/Food_{}/Annotations".format(canteen)
    with open(os.path.join(imgsets_path, "{}.txt").format(imgset), 'r') as f:
        xml_files = [x.strip("\n")+'.xml' for x in f.readlines()]
        content = []
        for xf in xml_files:
            objects = parse_rec(os.path.join(anno_path, xf))
            for obj in objects:
                # only reserve the !!! training sample whose count is larger than 10
                if N != 0:
                    match_categories = get_categories(
                        canteen+"_train_mt{}".format(N))
                else:
                    match_categories = get_categories(canteen+"_train")

                if obj['name'] in match_categories:
                    content.append(xf.split(".")[0] + '\n')
                    break

        print("saving {} sets:{}_mt{}".format(imgset, len(content), N))
        with open(os.path.join(imgsets_path, "{}mt{}.txt".format(imgset, N)), 'w') as f:
            f.writelines(content)


def create_trainval_imagesets(path: str):
    """create_trainval_imagesets

    :param path: path of dataset
    :type path: str

    """
    anno_path = os.path.join(path, "Annotations")
    imsets_path = os.path.join(path, "ImageSets")
    xml_files = sorted(os.listdir(anno_path))
    with open(os.path.join(imsets_path, "trainval.txt"), 'w') as f:
        f.writelines([x.split(".")[0]+'\n' for x in xml_files])


def create_all_canteen_train_and_val_imageset():
    canteen = ['Arts', 'Science', 'TechMixedVeg',
               'TechChicken', 'UTown', 'YIH']
    excl_canteen = ["excl"+x for x in canteen]
    canteen += excl_canteen
    canteen += ["All"]

    for ct in canteen:
        dishes_of_ct = create_dishes(ct)
        train_content, val_content = split_dishes2train_val(dishes_of_ct)
        create_train_and_val_imagesets(ct, train_content, val_content)


def create_all_canteen_mtN_train_and_val_imageset(N):
    canteen = ['Arts', 'Science', 'TechMixedVeg',
               'TechChicken', 'UTown', 'YIH']
    excl_canteen = ["excl"+x for x in canteen]
    canteen += excl_canteen
    canteen += ["All"]

    for ct in canteen:
        for imgset in ['train', 'val']:
            create_mtN_imageset(ct, imgset, N)


if __name__ == '__main__':
    create_all_canteen_train_and_val_imageset()
    for N in [10, 30, 50, 100]:
       create_all_canteen_mtN_train_and_val_imageset(N)
    create_inner_imagesets()
    create_few_inner_for_train_val("YIH", 'innermt10', mtN=10, fewN=1)
    create_mtN_imageset("exclYIH","val" ,0)
    create_few_inner_for_train_val("Arts", 'innermt10', mtN=10, fewN=1)
