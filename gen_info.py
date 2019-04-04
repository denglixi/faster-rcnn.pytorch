#!/usr/bin/env python
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.prefood_res50 import PreResNet50
from model.faster_rcnn.prefood_res50_attention import PreResNet50Attention
from model.faster_rcnn.prefood_res50_2fc import PreResNet502Fc
from model.faster_rcnn.prefood_res50_hi import PreResNet50Hierarchy
from model.faster_rcnn.prefood_res50_hi_ca import PreResNet50HierarchyCasecade
from model.faster_rcnn.prefood_res50_attention_minus import PreResNet50AttentionMinus
from model.faster_rcnn.prefood_res50_attention_pos import PreResNet50AttentionPos
from datasets.food_category import get_categories
from datasets.id2name import id2chn, id2eng
from datasets.voc_eval import get_gt_recs
from datasets.sub2main import sub2main_dict

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

beehoonid2name = {'1': 'bee hoon', '2': 'fried noodles', '3': 'kway teow', '4': 'kway teow, yellow noodles mix', '5': 'rice', '51': 'fried rice', '7': 'hokkien mee', '8': 'maggie noodle', '9': 'Glutinous rice', '10': 'beehoon and noodle mix', '110': 'stir fry mee tai mak', '11': 'fried egg', '12': 'scrambled egg', '13': 'cabbage', '131': 'hairy gourd with egg', '14': 'french bean/long bean', '141': 'broccoli', '142': 'celery', '143': 'beansprout', '15': 'deep fried beancurd skin', '16':
                  'fried beancurd/taukwa', '17': 'taupok', '171': 'braised taupok', '18': 'Acar', '181': 'Stir fried eggplant', '19': 'cucumber', '21': 'luncheon meat', '22': 'hashbrown', '23': 'ngoh hiang', '24': 'begedil', '25': 'spring roll', '31': 'otah', '32': 'fish ball/sotong ball', '33': 'white, yellow fish fillet', '331': 'orange, red fish fillet', '34': 'fish cake', '341': 'ngoh hiang fish cake', '35': 'kuning fish (fried small fish)', '351': 'fried fish steak', '36': 'siew mai',
                  '41': 'hotdog/taiwan sausage', '42': 'seaweed chicken', '43': 'chicken nugget', '44': 'fried chicken / chicken wings', '441': 'fried chicken chopped up', '45': 'fried chicken cutlet (not ground meat)', '55': 'curry mixed veg', '551': 'curry chicken and potato', '61': 'ikan bilis', '62': 'chilli paste', '63': 'green chilli', '64': 'peanut', '65': 'Sweet Sauce', '66': 'red chilli chopped', '71': 'deep fried fish', '91': 'Butter cereal chicken', '92': 'fried wanton/ dumpling', '93':
                  'Vegetarian meat', '94': 'Fried onions', '95': 'Crabstick'}

schoolid2name = {
    "21": 'Milk', "2": 'Drinkable yogurt', "3": 'Rice', "4": 'Mixed rice', "5": 'Bread', "6": 'White bread', "7": 'Udon', "8": 'Fish', "9": 'Meat', "10": 'Salad', "11": 'Cherry tomatoes', "12": 'Soups', "13": 'Curry', "14": 'Spicy chili-flavored tofu', "15": 'Bibimbap', "16": 'Fried noodles', "17": 'Spaghetti', "18": 'Citrus', "19": 'Apple', "20": 'Cup desserts', "1": 'Other foods'}


def get_det_bbox_with_cls_of_img(all_boxes, img_idx):
    """get_det_bbox_with_cls_of_img

    :param all_boxes: det result
    :param img_idx: img idx in imageset.txt
    :return bboxes: N*6-dim matrix, N is number of bbox, 6 is [x1,y1,x2,y2,score,cls], the cls is the addtion information we get from this function
    """
    # get all box of img
    img_all_boxes = [b[img_idx] for b in all_boxes]
    # add cls_idx to box_cls
    bboxes = None
    for cls_idx_img, boxes_of_cls in enumerate(img_all_boxes):
        if len(boxes_of_cls) != 0:
            cls_cloumn = np.zeros(len(boxes_of_cls)) + cls_idx_img
            # img_all_boxes[cls_idx] = np.c_[boxes_of_cls, cls_cloumn]
            if bboxes is None:
                bboxes = np.c_[boxes_of_cls, cls_cloumn]
            else:
                bboxes = np.vstack(
                    (bboxes, np.c_[boxes_of_cls, cls_cloumn]))
    return bboxes


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--imgset', dest='imgset',
                        help='val, test',
                        default='val', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--test_cache', dest='test_cache',
                        action='store_true')
    parser.add_argument('--save_for_vis', dest='save_for_vis',
                        type=bool, default=False)
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def cal_overlap(gt_boxes_array, det_boxes):
    """cal_overlap
    :param gt_boxes_array: np array
    :param det_boxes: two dim
    """

    det_overlaps = np.zeros(len(det_boxes))
    if gt_boxes_array is None or len(gt_boxes_array) == 0:
        return det_overlaps
    boxes_array = gt_boxes_array
    for b_i, box in enumerate(det_boxes):
        ixmin = np.maximum(boxes_array[:, 0], box[0])
        iymin = np.maximum(boxes_array[:, 1], box[1])
        ixmax = np.minimum(boxes_array[:, 2], box[2])
        iymax = np.minimum(boxes_array[:, 3], box[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) + (boxes_array[:, 2] -
                                                                  boxes_array[:, 0] + 1.) * (boxes_array[:, 3] - boxes_array[:, 1] + 1.) - inters)

        overlaps = inters / uni
        try:
            if type(overlaps) == torch.Tensor:
                overlaps = overlaps.numpy()

            ovmax = np.max(overlaps)
        except:
            pdb.set_trace()

        det_overlaps[b_i] = ovmax
    #jmax = np.argmax(overlaps)
    return det_overlaps


def get_data2imdbval_dict(imgset):
    # create canttens

    assert imgset in ['val', 'test']
    collected_cts = ["Arts", "Science", "YIH",
                     "UTown", "TechChicken", "TechMixedVeg", "EconomicBeeHoon"]
    excl_cts = ["excl"+x for x in collected_cts]
    all_canteens = collected_cts + excl_cts + ['All']

    # basic setting

    # create dict{ dataset -> imdb_name }
    # 1. create dataset -> dataset_val
    data2imdbval_dict = {}

    for ct in all_canteens:
        dataset = "food{}".format(ct)
        imdbval_name = "food_{}_{}_{}_train".format(
            ct, imgset, ct)
        data2imdbval_dict[dataset] = imdbval_name

    for ct in all_canteens:
        for mtN in [0, 10]:
            if mtN == 0:
                ct_sp = imgset
            else:
                ct_sp = "{}mt{}".format(imgset, mtN)

            # datasets here only support mtN format
            dataset = "food{}mt{}".format(ct, mtN)
            imdbval_name = "food_{}_{}_{}_train_mt{}".format(
                ct, ct_sp, ct, mtN)
            data2imdbval_dict[dataset] = imdbval_name

    # 2. create excl_dataset -> dataset
    for ct in collected_cts:
        for mtN in [0, 10]:
            for fewN in [0, 1, 5]:
                if mtN == 0:
                    mtN_str = ""
                else:
                    mtN_str = "mt{}".format(mtN)

                if fewN == 0:
                    fewN_str = ""
                else:
                    fewN_str = "few{}".format(fewN)

                # datasets here only support mtN format
                dataset = "foodexcl{}{}_test{}{}".format(
                    ct, mtN_str, ct,  fewN_str)

                if fewN == 0:
                    imdbval_name = "food_{}_inner{}{}_excl{}_train_mt{}".format(
                        ct, mtN_str, imgset, ct, mtN)  # innermt10val or innermt10test
                else:
                    imdbval_name = "food_{}_inner{}{}val_excl{}_train_mt{}".format(
                        ct, fewN_str, mtN_str, ct, mtN)  # it not working anymore . it is like innerfew1mt10val: TODO spliting to val and test
                data2imdbval_dict[dataset] = imdbval_name
    # 3. create exclcanteen_finecanteenfewN -> canteenfewN
    for ct in collected_cts:
        for mtN in [10]:
            for fewN in [1, 5, 10]:
                dataset = "foodexcl{}mt{}_fine{}few{}_test{}few{}".format(
                    ct, mtN, ct, fewN, ct, fewN)
                imdbval_name = "food_{}_innerfew{}mt{}val_excl{}_train_mt{}".format(
                    ct, fewN, mtN, ct, mtN)
                data2imdbval_dict[dataset] = imdbval_name

    # 4. extra
    data2imdbval_dict['schoollunch'] = 'schoollunch_val'
    return data2imdbval_dict


def set_imdbval_name(args):
    data2imdbval_dict = get_data2imdbval_dict(args.imgset)
    args.imdbval_name = data2imdbval_dict[args.dataset]
    return args


def get_main_cls(sub_classes):
    main_classes = []
    for i in sub_classes:
        main_classes.append(sub2main_dict[i])
    main_classes = set(main_classes)
    main_classes = list(main_classes)
    return main_classes


def pred_boxes_regression(boxes, bbox_pred, scores, classes, cfg, args):
    """pred_boxes_regression

    :param boxes: rois from RPN
    :param bbox_pred: regression result of each bbox
    :param scores: score of bboxes
    :param classes: classes of result
    :param cfg: cfg
    :param args: args
    """
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(
                    1, -1, 4 * len(classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = torch.from_numpy(
            np.tile(boxes, (1, scores.shape[2]))).cuda()

    pred_boxes /= data[1][0][2].item()
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    return scores, pred_boxes


if __name__ == '__main__':

    cfg.TRAIN.USE_FLIPPED = False
    imdbval_name = 'food_All_trainval_All_train'
    imdb, roidb, ratio_list, ratio_index = combined_roidb(
        imdbval_name, False)
    imdb.competition_mode(on=True)
    import pdb
    pdb.set_trace()
    num_images = len(imdb.image_index)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=5,
                                             pin_memory=True)
    data_iter = iter(dataloader)
    # size, overlap,
    sizes = []
    overlaps = []
    for img_index in range(num_images):
        if img_index % 100 == 99:
            print(img_index)

        data = next(data_iter)
        im_data = data[0]
        im_info = data[1]
        gt_boxes = data[2]
        num_boxes = data[3]
        for gt in gt_boxes:
            size = (gt[3] - gt[1]) * (gt[2] - gt[0])
            sizes.append(size)
            overlaps.append(cal_overlap(gt_boxes, [gt]))
    pdb.set_trace()
