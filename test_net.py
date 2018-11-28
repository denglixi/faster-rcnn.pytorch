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
from datasets.food_category import get_categories
from datasets.id2name import id2chn, id2eng


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


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

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "food":
        args.imdb_name = "food_YIH_train"
        args.imdbval_name = "food_YIH_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "foodtechmixed":
        args.imdb_name = "food_Tech_train"
        args.imdbval_name = "food_YIH_occur_in_tech"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "foodAll":
        args.imdb_name = "food_All_train_All_train"
        args.imdbval_name = "food_All_val_All_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "foodexclArts":
        args.imdb_name = "food_exclArts_train_exclArts_train"
        args.imdbval_name = "food_exclArts_val_exclArts_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclYIH":
        args.imdb_name = "food_exclYIH_train_exclYIH_train"
        args.imdbval_name = "food_exclYIH_val_exclYIH_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclUTown":
        args.imdb_name = "food_exclUTown_train_exclUTown_train"
        args.imdbval_name = "food_exclUTown_val_exclUTown_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclTechChicken":
        args.imdb_name = "food_exclTechChicken_train_exclTechChicken_train"
        args.imdbval_name = "food_exclTechChicken_val_exclTechChicken_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclTechMixedVeg":
        args.imdb_name = "food_exclTechMixedVeg_train_exclTechMixedVeg_train"
        args.imdbval_name = "food_exclTechMixedVeg_val_exclTechMixedVeg_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclScience":
        args.imdb_name = "food_exclScience_train_exclScience_train"
        args.imdbval_name = "food_exclScience_val_exclScience_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "foodexclArts_testArts":
        args.imdb_name = "food_exclArts_train_exclArts_train"
        args.imdbval_name = "food_Arts_inner_exclArts_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclYIH_testYIH":
        args.imdb_name = "food_exclYIH_train_exclYIH_train"
        args.imdbval_name = "food_YIH_inner_exclYIH_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclUTown_testUTown":
        args.imdb_name = "food_exclUTown_train_exclUTown_train"
        args.imdbval_name = "food_UTown_inner_exclUTown_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclTechChicken_testTechChicken":
        args.imdb_name = "food_exclTechChicken_train_exclTechChicken_train"
        args.imdbval_name = "food_TechChicken_inner_exclTechChicken_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclTechMixedVeg_testTechMixedVeg":
        args.imdb_name = "food_exclTechMixedVeg_train_exclTechMixedVeg_train"
        args.imdbval_name = "food_TechMixedVeg_inner_exclTechMixedVeg_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclScience_testScience":
        args.imdb_name = "food_exclScience_train_exclScience_train"
        args.imdbval_name = "food_Science_inner_exclScience_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodAllmt10":
        args.imdb_name = "food_All_trainmt10_All_train_mt10"
        args.imdbval_name = "food_All_valmt10_All_train_mt10"
        args.train_cls = "All_train_mt10"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodAllmt100":
        args.imdb_name = "food_All_trainmt10_All_train_mt10"
        args.imdbval_name = "food_All_valmt10_All_train_mt100"
        args.train_cls = "All_train_mt100"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodAllmt50":
        args.imdb_name = "food_All_trainmt10_All_train_mt10"
        args.imdbval_name = "food_All_valmt10_All_train_mt50"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclYIHmt10":
        args.imdb_name = "food_All_trainmt10_All_train_mt10"
        args.imdbval_name = "food_exclYIH_valmt10_exclYIH_train_mt10"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclYIHmt10_testYIH":
        args.imdb_name = "food_All_trainmt10_All_train_mt10"
        args.imdbval_name = "food_YIH_innermt10_exclYIH_train_mt10"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclUTownmt10_testUTown":
        args.imdb_name = "food_All_trainmt10_All_train_mt10"
        args.imdbval_name = "food_UTown_innermt10_exclUTown_train_mt10"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclUTownmt10":
        args.imdb_name = "food_All_trainmt10_All_train_mt10"
        args.imdbval_name = "food_exclUTown_val_exclUTown_train_mt10"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclYIHmt10_testYIHfew1":
        args.imdb_name = "food_All_trainfew10mt10_All_train_mt10"
        args.imdbval_name = "food_YIH_innerfew1mt10val_exclYIH_train_mt10"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclYIH_fineYIH_testYIHfew1":
        args.imdb_name = "food_All_trainfew10mt10_All_train_mt10"
        args.imdbval_name = "food_YIH_innerfew1mt10val_exclYIH_train_mt10"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclYIH_fineYIHfew5_testYIHfew5":
        args.imdb_name = "food_All_trainfew10mt10_All_train_mt10"
        args.imdbval_name = "food_YIH_innerfew5mt10val_exclYIH_train_mt10"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "foodexclYIH_testYIHfew1":
        args.imdb_name = "food_All_trainfew10mt10_All_train_mt10"
        args.imdbval_name = "food_YIH_innerfew1mt10val_exclYIH_train_mt10"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    #args = construct_dataset(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(
        args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))
    if len(args.dataset.split('_')) > 1:
        model_name = "_".join(args.dataset.split('_')[0:-1])
    else:
        model_name = args.dataset
    input_dir = args.load_dir + "/" + args.net + \
        "/" + model_name
    if not os.path.exists(input_dir):
        raise Exception(
            'There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False,
                           class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'foodres50':
        fasterRCNN = PreResNet50(imdb.classes, pretrained=True,
                                 class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.0005
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_{}_{}_{}'.format(
        args.checksession, args.checkepoch, args.checkpoint)
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=5,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    if(args.test_cache):
        with open(det_file, 'rb') as f:
            all_boxes = pickle.load(f)
        #imdb.evaluate_detections(all_boxes, output_dir)
        # exit()
        # if args.save_for_vis:
        #    gt_roidb = imdb.gt_roidb()
        #    for im_idx in len(imdb.image_index):
        #        pass
        #        # Not Implement
        #        # TODO 1. save cls that was not been detected or was wrong classified
        #        # TODO 1. save cls that wrong detected
        #        #

    else:
        fasterRCNN.eval()
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
        for i in range(num_images):
            data_tic = time.time()
            data = next(data_iter)
            # if imdb.image_index[i] == '11oct_DONE328IMG_20181011_115438':
            #    pass
            #    #import pdb
            #    #pdb.set_trace()
            # else:
            #    continue
            data_toc = time.time()
            data_load_time = data_toc - data_tic
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            # filter boxes with lower score
            # It is 0 for batch size is 1
            gt_boxes_cpu = gt_boxes.cpu().numpy()[0]
            gt_boxes_cpu[:, 0:4] /= float(im_info[0][2].cpu().numpy())

            if args.save_for_vis:
                save_vis_root_path = './savevis/{}_{}_{}/'.format(
                    args.checksession, args.checkepoch, args.checkpoint)

                # if vis or args.save_for_vis:
                #    im = cv2.imread(imdb.image_path_at(i))
                #    im2show = np.copy(im)

                # show ground-truth
                # for gt_b in gt_boxes_cpu:
                #    im2show = vis_detections(
                #        im2show, id2chn[imdb.classes[int(gt_b[-1])]], gt_b[np.newaxis, :], 0.1, (204, 0, 0))

            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

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
                            1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = torch.from_numpy(
                    np.tile(boxes, (1, scores.shape[2]))).cuda()

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()

            detect_time = det_toc - det_tic
            misc_tic = time.time()
            if vis or args.save_for_vis:
                im = cv2.imread(imdb.image_path_at(i))
                im2show = np.copy(im)
            for j in xrange(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat(
                        (cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis or args.save_for_vis:
                        im2show = vis_detections(
                            im2show, id2chn[imdb.classes[j]], np.array([cls_dets.cpu().numpy()[0, :]]), 0.5)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array
            # Limit to max_per_image detections *over all classes*
            #max_per_image = 1
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in xrange(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(
                            all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            # save images by gt for analysis
            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} dataload:{:.3f}s det:{:.3f}s nms:{:.3f}s  \r'
                             .format(i + 1, num_images, data_load_time,  detect_time, nms_time))
            sys.stdout.flush()

            if vis:
                # cv2.imwrite('result.png', im2show)
                # pdb.set_trace()
                cv2.namedWindow("frame", 0)
                cv2.resizeWindow("frame", 800, 800)
                cv2.imshow('frame', im2show)
                cv2.waitKey(0)

            # To save image for analysis
            # Limit to threshhold detections *over all classes*
            threshold_of_vis = 0.1
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in xrange(1, imdb.num_classes)])
                # np.sort(image_scores)[-max_per_image]
                image_thresh = threshold_of_vis
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

            if args.save_for_vis:
                boxes_of_i = np.array([_[i] for _ in all_boxes])

                # filter boxes with lower score
                # It is 0 for batch size is 1
                gt_boxes_cpu = gt_boxes.cpu().numpy()[0]
                gt_boxes_cpu[:, 0:4] /= float(im_info[0][2].cpu().numpy())

                save_vis_root_path = './savevis/{}_{}_{}/'.format(
                    args.checksession, args.checkepoch, args.checkpoint)

                # show ground-truth
                for gt_b in gt_boxes_cpu:
                    im2show = vis_detections(
                        im2show, id2chn[imdb.classes[int(gt_b[-1])]], gt_b[np.newaxis, :], 0.1, (204, 0, 0))

                i_row, i_c, _ = im2show.shape
                im2show = cv2.resize(im2show, (int(i_c/2), int(i_row/2)))

                # 1.gt未检测到
                # 2. gt类别错误(TODO)
                for gt_b in gt_boxes_cpu:
                    gt_cls_idx = int(gt_b[4])
                    # 1 && 2
                    if len(boxes_of_i[gt_cls_idx]) == 0:
                        save_vis_path = save_vis_root_path + \
                            'FN/' + id2chn[imdb.classes[int(gt_cls_idx)]]
                        if not os.path.exists(save_vis_path):
                            os.makedirs(save_vis_path)
                        # im2vis_analysis = vis_detections(
                        #    im2show, imdb.classes[int(gt_b[-1])], gt_b[np.newaxis,:], 0.1, (204, 0, 0))
                        cv2.imwrite(os.path.join(save_vis_path,
                                                 imdb.image_index[i]+'.jpg'), im2show)

                gt_classes = [int(_[-1]) for _ in gt_boxes_cpu]
                # 3. FP
                for bi, det_b_cls in enumerate(boxes_of_i):
                    if len(det_b_cls) > 0 and any(det_b_cls[:, 4] > 0.5):
                        if bi not in gt_classes:
                            save_vis_path = save_vis_root_path + \
                                'FP/' + id2chn[str(imdb.classes[bi])]
                            if not os.path.exists(save_vis_path):
                                os.makedirs(save_vis_path)
                            cv2.imwrite(os.path.join(save_vis_path,
                                                     imdb.image_index[i]+'.jpg'), im2show)

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')

    # save_map_result(imdb, all_boxes, output_dir, args):
    test_map_results = True
    if test_map_results:
        cls_ap_zip, dataset_map = imdb.evaluate_detections(
            all_boxes, output_dir)
        # results_filename = args.imdbval_name + \
        #    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        results_filename = args.imdbval_name + "_session" + \
            str(args.checksession) + "_epoch" + str(args.checkepoch)
        results_save_dir = "test_result/"
        if not os.path.exists(results_save_dir):
            os.makedirs(results_save_dir)

        # filter results
        #train_cls = get_categories(args.train_cls)
        val_names = args.imdbval_name.split("_")
        val_canteen = val_names[1]  # YIH
        val_split = val_names[2]  # valmt10 -> val_mt10
        if 'inner' in val_split:
            val_categories = get_categories(val_canteen+"_"+"inner")
        else:
            val_categories, val_ap = zip(*cls_ap_zip)
            cls_ap_zip = zip(val_categories, val_ap)

        # f.write(str(dataset_map) + '\n')

        map_exist_cls = []
        with open(os.path.join(results_save_dir, results_filename), 'w') as f:
            for cls, ap in cls_ap_zip:
                if str(cls) in val_categories:
                    if np.isnan(ap):
                        f.write(str(cls) + '\n')
                    else:
                        f.write(str(cls) + '\t' + str(ap) + '\n')
                        map_exist_cls.append(ap)
            map_exist_cls = sum(map_exist_cls) / len(map_exist_cls)
            print("exist cls map:{}".format(map_exist_cls))
            f.write(str(map_exist_cls))

        end = time.time()
        print("test time: %0.4fs" % (end - start))

    test_cls_loc = True
    threshold = 0.9
    if test_cls_loc:
        loc_accuracy, clsify_accuracy = imdb.evaluate_cls_loc(
            all_boxes, threshold)
        #acs = []
        # for cls, ac in loc_accuracy:
        #    if not np.isnan(ac):
        #        acs.append(ac)
        # print('loc_accuracy')
        # print(loc_accuracy)
        # print(np.mean(acs))
        # for cls, ac in clsify_accuracy:
        #    if not np.isnan(ac):
        #        acs.append(ac)
        # print('clsify_accuracy')
        # print(clsify_accuracy)
        # print(np.mean(acs))

        #
        for metrics in ['loc', 'cls']:
            if metrics == 'loc':
                itertor = loc_accuracy
            elif metrics == 'cls':
                itertor = clsify_accuracy

            results_filename = args.imdbval_name + "_session" + \
                str(args.checksession) + "_epoch" + \
                str(args.checkepoch) + "{}".format(metrics)
            results_save_dir = "test_result/"
            if not os.path.exists(results_save_dir):
                os.makedirs(results_save_dir)

            # filter results
            # train_cls = get_categories(args.train_cls)
            val_names = args.imdbval_name.split("_")
            val_canteen = val_names[1]  # YIH
            val_split = val_names[2]  # valmt10 -> val_mt10
            if 'inner' in val_split:
                val_categories = get_categories(val_canteen+"_"+"inner")
            else:
                val_categories = [x[0] for x in itertor]

            # f.write(str(dataset_map) + '\n')

            map_exist_cls = []
            with open(os.path.join(results_save_dir, results_filename), 'w') as f:
                for cls, ap in itertor:
                    if str(cls) in val_categories:
                        if np.isnan(ap):
                            f.write(str(cls) + '\n')
                        else:
                            f.write(str(cls) + '\t' + str(ap) + '\n')
                            map_exist_cls.append(ap)
                map_exist_cls = sum(map_exist_cls) / len(map_exist_cls)
                print("exist cls map:{}".format(map_exist_cls))
                f.write(str(map_exist_cls))

            end = time.time()
            print("test time: %0.4fs" % (end - start))
