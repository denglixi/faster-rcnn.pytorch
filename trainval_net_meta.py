# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb_meta
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, set_learning_rate

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.prefood_res50 import PreResNet50
from model.faster_rcnn.prefood_res50_hi import PreResNet50Hierarchy
from model.faster_rcnn.prefood_res50_hi_ca import PreResNet50HierarchyCasecade
from model.faster_rcnn.prefood_res50_2fc import PreResNet502Fc
from model.faster_rcnn.prefood_res50_meta import PreResNet50Meta
from model.faster_rcnn.prefood_res50_attention import PreResNet50Attention

from model.utils.net_utils import vis_detections
from datasets.id2name import id2chn, id2eng
from datasets.sub2main import sub2main_dict


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101, food_res50, food_res50_hierarchy',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

# config optimization
    parser.add_argument('--wu', dest='warming_up',
                        help='train from scrach',
                        action='store_true')
    parser.add_argument('--wulr', dest='warming_up_lr',
                        help='learning rate at 1st epoch if set up warming_up',
                        default=0.0001, type=float)
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.3, type=float)

# set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

# resume trained model
    parser.add_argument('--pretrained', dest='pretrained',
                        help='use pretrained model or not',
                        default=True, type=bool)
    parser.add_argument('--fixed_layer', dest='fixed_layer',
                        help='determin the fixed layer of resnet50',
                        default=0, type=int)
    parser.add_argument('--weight_file', dest='weight_file',
                        help='imagenet, prefood',
                        default='vgg16', type=str)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--resume_session_epoch', dest='resume_session_epoch',
                        help='resume session and epoch or not',
                        default=False, type=bool)
    parser.add_argument('--resume_opt', dest='resume_opt',
                        help='resume optimizer or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
# log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
# save model
    parser.add_argument('--save_model', dest='save_model',
                        help="whether save model",
                        action='store_true')
    parser.add_argument('--save_epoch', dest='save_epoch',
                        help='save model per save_epoch epoch',
                        default=1, type=int)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(
                self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(
            self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(
            self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat(
                (self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def get_data2imdb_dict():
    # create canttens
    collected_cts = ["Arts", "Science", "YIH",
                     "UTown", "TechChicken", "TechMixedVeg"]
    excl_cts = ["excl"+x for x in collected_cts]
    all_canteens = collected_cts + excl_cts + ['All']

    # create dict{ dataset -> imdb_name }
    data2imdb_dict = {}

    # 1. train on origin mt
    for ct in all_canteens:
        for mtN in [0, 10]:
            if mtN == 0:
                ct_sp = "train"
            else:
                ct_sp = "trainmt{}".format(mtN)

            imdb_name = "food_{}_{}_{}_train_mt{}".format(ct, ct_sp, ct, mtN)
            dataset = "food{}mt{}".format(ct, mtN)
            data2imdb_dict[dataset] = imdb_name

    # 2. trian on fine
    for ct in collected_cts:
        for mtN in [10]:
            for fewN in [1, 5, 10]:
                dataset = "foodexcl{}mt{}_fine{}few{}".format(
                    ct, mtN, ct, fewN)
                imdb_name = "food_{}_innerfew{}mt{}train_excl{}_train_mt{}".format(
                    ct, fewN, mtN, ct, mtN)
                data2imdb_dict[dataset] = imdb_name

    for ct in collected_cts:
        dataset = "food_meta_{}_train".format(ct)
        imdb_name = dataset
        data2imdb_dict[dataset] = imdb_name
    return data2imdb_dict


def set_imdb_name(args):
    data2imdb_dict = get_data2imdb_dict()
    args.imdb_name = data2imdb_dict[args.dataset]
    return args


def get_main_cls(sub_classes):
    main_classes = []
    for i in sub_classes:
        main_classes.append(sub2main_dict[i])
    main_classes = set(main_classes)
    main_classes = list(main_classes)
    return main_classes


def test_rotate():
    # from collections import defaultdict
    vis_dict = {}
    for fli in [0, 1]:
        for angle in [0, 90, 180, 270]:
            vis_dict[fli+angle] = 0
    pass
    # visual test for rotated
    # flipped = data[4].numpy()[0]
    # rotated = data[5].numpy()[0]

    # if vis_dict[flipped+rotated] == 0:
    #    gt_boxes_cpu = gt_boxes.cpu().numpy()[0]
    #    #gt_boxes_cpu[:, 0:4] /= float(im_info[0][2].cpu().numpy())

    #    im2show = np.transpose(im_data.cpu().numpy()[
    #                           0], (1, 2, 0)).astype(np.uint8)
    #    #im2show = im2show[:, :, ::-1]
    #    im2show = im2show.copy()
    #    for gt_b in gt_boxes_cpu[np.where(gt_boxes_cpu[:,4]>0)]:
    #        im2show = vis_detections(
    #            im2show, id2chn[imdb.classes[int(gt_b[-1])]], gt_b[np.newaxis, :], 0.1, (204, 0, 0))
    #    cv2.imshow("flipped{}_rotated{}".format(flipped, rotated), im2show)
    #    cv2.waitKey()
    #    vis_dict[flipped+rotated] = 1
    # else:
    #    continue


def init_network(args):

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=args.pretrained,
                           class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=args.pretrained,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=args.pretrained,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=args.pretrained,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'foodres50':
        fasterRCNN = PreResNet50(imdb.classes, pretrained=args.pretrained,
                                 class_agnostic=args.class_agnostic,
                                 weight_file=args.weight_file,
                                 fixed_layer=args.fixed_layer)

    elif args.net == 'foodres50attention':
        fasterRCNN = PreResNet50Attention(imdb.classes, pretrained=args.pretrained,
                                          class_agnostic=args.class_agnostic,
                                          weight_file=args.weight_file,
                                          fixed_layer=args.fixed_layer)

    elif args.net == 'foodres502fc':
        fasterRCNN = PreResNet502Fc(imdb.classes, pretrained=args.pretrained,
                                    class_agnostic=args.class_agnostic,
                                    weight_file=args.weight_file,
                                    fixed_layer=args.fixed_layer)

    elif args.net == 'foodres50meta':
        fasterRCNN = PreResNet50Meta([0 for _ in range(170)], pretrained=args.pretrained,
                                     class_agnostic=args.class_agnostic,
                                     weight_file=args.weight_file,
                                     fixed_layer=args.fixed_layer)

    elif args.net == 'foodres50_hierarchy':
        main_classes = get_main_cls(imdb.classes)
        fasterRCNN = PreResNet50Hierarchy(main_classes, imdb.classes,
                                          pretrained=args.pretrained,
                                          class_agnostic=args.class_agnostic,
                                          weight_file=args.weight_file,
                                          fixed_layer=args.fixed_layer)
    elif 'foodres50_hierarchy_casecade' in args.net:
        nets_param = args.net.split('_')
        if len(nets_param) == 3:
            casecade_type = 'add_score'
            alpha = 0.5
        elif len(nets_param) == 5:
            casecade_type = "".join(nets_param[3:5])
            alpha = 0.5
        elif len(nets_param) == 6:
            casecade_type = "".join(nets_param[3:5])
            alpha = float(nets_param[5])

        main_classes = get_main_cls(imdb.classes)
        fasterRCNN = PreResNet50HierarchyCasecade(main_classes, imdb.classes,
                                                  pretrained=args.pretrained,
                                                  class_agnostic=args.class_agnostic,
                                                  weight_file=args.weight_file,
                                                  fixed_layer=args.fixed_layer,
                                                  casecade_type=casecade_type,
                                                  alpha=alpha)
    else:
        print("network is not defined")
        pdb.set_trace()

    return fasterRCNN


def forward_faster(fasterRCNN, im_data, im_info, gt_boxes, num_boxes, loss_temp, weights=None):
    if weights is None:
        if "hierarchy" in args.net:
            rois, cls_prob_main, bbox_pred_main, \
                cls_prob_sub, bbox_pred_sub, \
                rpn_loss_cls, rpn_loss_box, \
                *RCNN_losses, \
                rois_label = fasterRCNN(
                    im_data, im_info, gt_boxes, num_boxes)
        else:
            rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                *RCNN_losses, \
                rois_label = fasterRCNN(
                    im_data, im_info, gt_boxes, num_boxes)
    else:
        RCNN_top_fast_weight = weights[0]
        RCNN_cls_score_weight = weights[1]
        RCNN_bbox_pred_weight = weights[2]
        if "hierarchy" in args.net:
            rois, cls_prob_main, bbox_pred_main, \
                cls_prob_sub, bbox_pred_sub, \
                rpn_loss_cls, rpn_loss_box, \
                *RCNN_losses, \
                rois_label = fasterRCNN(
                    im_data, im_info, gt_boxes, num_boxes, RCNN_top_fast_weight, RCNN_cls_score_weight, RCNN_bbox_pred_weight)
        else:
            rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                *RCNN_losses, \
                rois_label = fasterRCNN(
                    im_data, im_info, gt_boxes, num_boxes, RCNN_top_fast_weight, RCNN_cls_score_weight, RCNN_bbox_pred_weight)

    if len(RCNN_losses) == 2:
        RCNN_loss_cls = RCNN_losses[0]
        RCNN_loss_bbox = RCNN_losses[1]
        RCNN_loss_cls_main = None
        RCNN_loss_bbox_main = None
    elif len(RCNN_losses) == 4:
        RCNN_loss_cls = RCNN_losses[0]
        RCNN_loss_bbox = RCNN_losses[1]
        RCNN_loss_cls_main = RCNN_losses[2]
        RCNN_loss_bbox_main = RCNN_losses[3]

    loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
    loss_temp += loss.item()
    return loss, loss_temp


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    args = set_imdb_name(args)

    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                     'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.cuda:
        cfg.CUDA = True

    fasterRCNN = init_network(args)
    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr':lr,
                            'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        # lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        # resume_dir = args.save_dir + "/" + \
        #    args.net + "/" + args.dataset.split("_")[0]
        resume_dir = './models/foodres50/foodexclYIHmt10/'
        load_name = os.path.join(resume_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession,
                                                                   args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        if args.resume_session_epoch:
            args.session = checkpoint['session']
            args.start_epoch = checkpoint['epoch']
        param_dict = checkpoint['model']
        param_dict.pop('RCNN_cls_score.weight')
        param_dict.pop('RCNN_cls_score.bias')
        param_dict.pop('RCNN_bbox_pred.weight')
        param_dict.pop('RCNN_bbox_pred.bias')
        fasterRCNN.load_state_dict(param_dict, strict=False)
        if args.resume_opt:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
        # if 'pooling_mode' in checkpoint.keys():
        #    cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    # if args.mGPUs:
    #    fasterRCNN = nn.DataParallel(fasterRCNN)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs/" + args.dataset +
                               "_" + str(args.session))

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    im_data_meta = torch.FloatTensor(1)
    im_info_meta = torch.FloatTensor(1)
    num_boxes_meta = torch.LongTensor(1)
    gt_boxes_meta = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        im_data_meta = im_data_meta.cuda()
        im_info_meta = im_info_meta.cuda()
        num_boxes_meta = num_boxes_meta.cuda()
        gt_boxes_meta = gt_boxes_meta.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    im_data_meta = Variable(im_data_meta)
    im_info_meta = Variable(im_info_meta)
    num_boxes_meta = Variable(num_boxes_meta)
    gt_boxes_meta = Variable(gt_boxes_meta)

    num_of_task = 10
    for epoch in range(num_of_task):
        print("task {}".format(epoch))
        # dataset
        imdb, roidb, ratio_list, ratio_index, imdb_test, roidb_test, ratio_list_test, ratio_index_test = combined_roidb_meta(
            args.imdb_name)

        base_train_size = len(roidb)
        meta_train_size = len(roidb_test)

        iters_per_base = int(base_train_size / args.batch_size)
        iters_per_meta = int(meta_train_size / args.batch_size)

        print('{:d} roidb entries'.format(len(roidb)))
        # base data
        sampler_batch = sampler(base_train_size, args.batch_size)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                                 imdb.num_classes, training=True)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 sampler=sampler_batch,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)

        # meta data
        sampler_batch_meta = sampler(meta_train_size, args.batch_size)
        dataset_meta = roibatchLoader(roidb_test, ratio_list_test, ratio_index_test,
                                      args.batch_size,
                                      imdb_test.num_classes,
                                      training=True)

        dataloader_meta = torch.utils.data.DataLoader(dataset_meta,
                                                      batch_size=args.batch_size,
                                                      sampler=sampler_batch_meta,
                                                      num_workers=args.num_workers,
                                                      pin_memory=True)

        # setting to train mode
        fasterRCNN.train()
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        loss_temp = 0
        data_iter = iter(dataloader)
        update_lr = 0.001

        # first step in base training

        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        fasterRCNN.zero_grad()
        loss, loss_temp = forward_faster(
            fasterRCNN, im_data, im_info, gt_boxes, num_boxes, loss_temp)
        print('loss', loss_temp)

        # update fast weight
        grad_RPN = torch.autograd.grad(
            loss, fasterRCNN.RCNN_rpn.parameters(), retain_graph=True)
        grad_RCNN_top = []
        for n, p in fasterRCNN.RCNN_top.named_parameters():
            if p.requires_grad:
                grad = torch.autograd.grad(loss, p, retain_graph=True)
                if len(grad) != 1:
                    pdb.set_trace()

                grad_RCNN_top.append(grad[0])

        grad_RCNN_cls = torch.autograd.grad(
            loss, fasterRCNN.RCNN_cls_score.parameters(), retain_graph=True)
        grad_RCNN_reg = torch.autograd.grad(
            loss, fasterRCNN.RCNN_bbox_pred.parameters(), retain_graph=False)

        RCNN_top_params = []
        for n, p in fasterRCNN.RCNN_top.named_parameters():
            if p.requires_grad:
                RCNN_top_params.append(p)

        RCNN_top_fast_weight = list(
            map(lambda p: p[1] - update_lr * p[0], zip(grad_RCNN_top, RCNN_top_params)))

        RCNN_cls_score_weight = list(map(
            lambda p: p[1] - update_lr * p[0], zip(grad_RCNN_cls, fasterRCNN.RCNN_cls_score.parameters())))

        RCNN_bbox_pred_weight = list(map(
            lambda p: p[1] - update_lr * p[0], zip(grad_RCNN_reg, fasterRCNN.RCNN_bbox_pred.parameters())))

        loss_meta = 0

        for base_update_epoch in range(5):
            data_iter = iter(dataloader)
            for update_base_step in range(1, iters_per_base):
                if(update_base_step % 20 == 0):
                    print('loss', loss_temp / 20)
                    loss_temp = 0

                data = next(data_iter)
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])

                loss, loss_temp = forward_faster(fasterRCNN,
                                                 im_data, im_info, gt_boxes, num_boxes,
                                                 loss_temp,
                                                 [
                                                     RCNN_top_fast_weight,
                                                     RCNN_cls_score_weight,
                                                     RCNN_bbox_pred_weight
                                                 ])

                # update fast weight
                # 1. cal grad
                grad_RPN = torch.autograd.grad(
                    loss, fasterRCNN.RCNN_rpn.parameters(), retain_graph=True)
                grad_RCNN_top = []
                for n, p in fasterRCNN.RCNN_top.named_parameters():
                    if p.requires_grad == True:
                        grad = torch.autograd.grad(loss, p, retain_graph=True)
                        grad_RCNN_top.append(grad[0])

                grad_RCNN_cls = torch.autograd.grad(
                    loss, fasterRCNN.RCNN_cls_score.parameters(), retain_graph=True)
                grad_RCNN_reg = torch.autograd.grad(
                    loss, fasterRCNN.RCNN_bbox_pred.parameters(), retain_graph=False)

                update_lr = 0.01
                # RPN_fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad_RPN, fasterRCNN.RCNN_rpn.parameters())

                # 2. update fast weight
                RCNN_top_params = []
                for n, p in fasterRCNN.RCNN_top.named_parameters():
                    if p.requires_grad == True:
                        RCNN_top_params.append(p)

                RCNN_top_fast_weight = list(
                    map(lambda p: p[1] - update_lr * p[0], zip(grad_RCNN_top, RCNN_top_fast_weight)))

                RCNN_cls_score_weight = list(map(
                    lambda p: p[1] - update_lr * p[0], zip(grad_RCNN_cls, RCNN_cls_score_weight)))

                RCNN_bbox_pred_weight = list(map(
                    lambda p: p[1] - update_lr * p[0], zip(grad_RCNN_reg, RCNN_bbox_pred_weight)))

        data_iter_meta = iter(dataloader_meta)
        for update_meta_step in range(iters_per_meta):
            data_meta = next(data_iter_meta)
            im_data_meta.data.resize_(data_meta[0].size()).copy_(data_meta[0])
            im_info_meta.data.resize_(data_meta[1].size()).copy_(data_meta[1])
            gt_boxes_meta.data.resize_(data_meta[2].size()).copy_(data_meta[2])
            num_boxes_meta.data.resize_(
                data_meta[3].size()).copy_(data_meta[3])

            loss_meta_taski, loss_temp = forward_faster(fasterRCNN,
                                                        im_data_meta, im_info_meta,
                                                        gt_boxes_meta, num_boxes_meta,
                                                        loss_temp,
                                                        [
                                                            RCNN_top_fast_weight,
                                                            RCNN_cls_score_weight,
                                                            RCNN_bbox_pred_weight
                                                        ])
            loss_meta += loss_meta_taski
            print("loss meta:{}".format(loss_meta))

            optimizer.zero_grad()
            loss_meta.backward()
            optimizer.step()
            pdb.set_trace()
        print(sum(sum(list(fasterRCNN.RCNN_cls_score.parameters())[0])))

        pdb.set_trace()

        save_name = os.path.join(
            output_dir, 'faster_rcnn_{}.pth'.format(args.session))
        if args.save_model:
        #if (epoch+1) % args.save_epoch == 0:
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                #'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                'model': fasterRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
            print('save model: {}'.format(save_name))
    # if args.use_tfboard:
    #    logger.close()
