#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 denglixi <denglixi@xgpd1>
#
# Distributed under terms of the MIT license.

"""

"""
with open("./engname.txt", 'r') as f:
    names = [x.strip('\n') for x in f.readlines()]
with open("id.py", 'a') as f:
    f.write("engname=[")
    for n in names:
        f.write('\'' + n + '\', ')
    f.write("]\n")
