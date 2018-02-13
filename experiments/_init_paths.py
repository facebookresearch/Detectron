#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 08:39:36 2018

@author: roy
"""

"""Insert /home/roy/projects/caffe2/build to PYTHONPATH"""

import sys

pt = '/home/roy/projects/caffe2/build'


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
add_path(pt)