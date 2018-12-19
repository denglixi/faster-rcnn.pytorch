#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 denglixi <denglixi@xgpd0>
#
# Distributed under terms of the MIT license.

"""

"""
with open('./id.txt') as f:
    sub_classes = [x.strip() for x in f.readlines()]
with open('./maincls.txt') as f:
    main_classes = [x.strip() for x in f.readlines()]
with open('sub2main.py','w') as f:
    f.write('{')
    for i in range(len(sub_classes)):
        f.write('\'{}\':\'{}\','.format(sub_classes[i], main_classes[i]))
    f.write('}')
