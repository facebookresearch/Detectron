# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Handle mapping from old network building function names to new names.

Flexible network configuration is achieved by specifying the function name that
builds a network module (e.g., the name of the conv backbone or the mask roi
head). However we may wish to change names over time without breaking previous
config files. This module provides backwards naming compatibility by providing
a mapping from the old name to the new name.

When renaming functions, it's generally a good idea to codemod existing yaml
config files. An easy way to batch edit, by example, is a shell command like

$ find . -name "*.yaml" -exec sed -i -e \
   's/head_builder\.add_roi_2mlp_head/fast_rcnn_heads.add_roi_2mlp_head/g' {} \;

to perform the renaming:
  head_builder.add_roi_2mlp_head => fast_rcnn_heads.add_roi_2mlp_head
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


_RENAME = {
    # Removed "ResNet_" from the name because it wasn't relevent
    'mask_rcnn_heads.ResNet_mask_rcnn_fcn_head_v1up4convs':
        'mask_rcnn_heads.mask_rcnn_fcn_head_v1up4convs',
    # Removed "ResNet_" from the name because it wasn't relevent
    'mask_rcnn_heads.ResNet_mask_rcnn_fcn_head_v1up':
        'mask_rcnn_heads.mask_rcnn_fcn_head_v1up',
    # Removed "ResNet_" from the name because it wasn't relevent
    'mask_rcnn_heads.ResNet_mask_rcnn_fcn_head_v0upshare':
        'mask_rcnn_heads.mask_rcnn_fcn_head_v0upshare',
    # Removed "ResNet_" from the name because it wasn't relevent
    'mask_rcnn_heads.ResNet_mask_rcnn_fcn_head_v0up':
        'mask_rcnn_heads.mask_rcnn_fcn_head_v0up',
    # Removed head_builder module in favor of the more specific fast_rcnn name
    'head_builder.add_roi_2mlp_head':
        'fast_rcnn_heads.add_roi_2mlp_head',
}


def get_new_name(func_name):
    if func_name in _RENAME:
        func_name = _RENAME[func_name]
    return func_name
