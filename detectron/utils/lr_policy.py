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

"""Learning rate policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from core.config import cfg


def get_lr_at_iter(it):
    """Get the learning rate at iteration it according to the cfg.SOLVER
    settings.
    """
    lr = get_lr_func()(it)
    if it < cfg.SOLVER.WARM_UP_ITERS:
        method = cfg.SOLVER.WARM_UP_METHOD
        if method == 'constant':
            warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
        elif method == 'linear':
            alpha = it / cfg.SOLVER.WARM_UP_ITERS
            warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
        else:
            raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
        lr *= warmup_factor
    return np.float32(lr)


# ---------------------------------------------------------------------------- #
# Learning rate policy functions
# ---------------------------------------------------------------------------- #

def lr_func_steps_with_lrs(cur_iter):
    """For cfg.SOLVER.LR_POLICY = 'steps_with_lrs'

    Change the learning rate to specified values at specified iterations.

    Example:
    cfg.SOLVER.MAX_ITER: 90
    cfg.SOLVER.STEPS:    [0,    60,    80]
    cfg.SOLVER.LRS:      [0.02, 0.002, 0.0002]
    for cur_iter in [0, 59]   use 0.02
                 in [60, 79]  use 0.002
                 in [80, inf] use 0.0002
    """
    ind = get_step_index(cur_iter)
    return cfg.SOLVER.LRS[ind]


def lr_func_steps_with_decay(cur_iter):
    """For cfg.SOLVER.LR_POLICY = 'steps_with_decay'

    Change the learning rate specified iterations based on the formula
    lr = base_lr * gamma ** lr_step_count.

    Example:
    cfg.SOLVER.MAX_ITER: 90
    cfg.SOLVER.STEPS:    [0,    60,    80]
    cfg.SOLVER.BASE_LR:  0.02
    cfg.SOLVER.GAMMA:    0.1
    for cur_iter in [0, 59]   use 0.02 = 0.02 * 0.1 ** 0
                 in [60, 79]  use 0.002 = 0.02 * 0.1 ** 1
                 in [80, inf] use 0.0002 = 0.02 * 0.1 ** 2
    """
    ind = get_step_index(cur_iter)
    return cfg.SOLVER.BASE_LR * cfg.SOLVER.GAMMA ** ind


def lr_func_step(cur_iter):
    """For cfg.SOLVER.LR_POLICY = 'step'
    """
    return (
        cfg.SOLVER.BASE_LR *
        cfg.SOLVER.GAMMA ** (cur_iter // cfg.SOLVER.STEP_SIZE))


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #

def get_step_index(cur_iter):
    """Given an iteration, find which learning rate step we're at."""
    assert cfg.SOLVER.STEPS[0] == 0, 'The first step should always start at 0.'
    steps = cfg.SOLVER.STEPS + [cfg.SOLVER.MAX_ITER]
    for ind, step in enumerate(steps):  # NoQA
        if cur_iter < step:
            break
    return ind - 1


def get_lr_func():
    policy = 'lr_func_' + cfg.SOLVER.LR_POLICY
    if policy not in globals():
        raise NotImplementedError(
            'Unknown LR policy: {}'.format(cfg.SOLVER.LR_POLICY))
    else:
        return globals()[policy]
