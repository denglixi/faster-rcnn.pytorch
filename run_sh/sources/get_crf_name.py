#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 denglixi <denglixi@xgpd7>
#
# Distributed under terms of the MIT license.

"""

"""
import _init_paths
import pdb
from datasets.id2name import id2chn, id2eng

with open('./co_cls.txt', 'r') as f:
    ids = [x.strip() for x in f.readlines()]

pdb.set_trace()

chn_names = [id2chn[x] for x in ids]
eng_names = [id2eng[x] for x in ids]


with open('./co_name.txt','w') as f:
    for i, c_n in enumerate(chn_names):
        f.write(c_n +'\t' + eng_names[i]+'\n')
