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

"""Primitives for running multiple single-GPU jobs in parallel over subranges of
data. These are used for running multi-GPU inference. Subprocesses are used to
avoid the GIL since inference may involve non-trivial amounts of Python code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import subprocess
from six.moves import shlex_quote

from detectron.core.config import cfg
from detectron.utils.io import load_object
import detectron.utils.env as envu

import logging
logger = logging.getLogger(__name__)


def process_in_parallel(
    tag, total_range_size, binary, output_dir, opts=''
):
    """Run the specified binary cfg.NUM_GPUS times in parallel, each time as a
    subprocess that uses one GPU. The binary must accept the command line
    arguments `--range {start} {end}` that specify a data processing range.
    """
    # Snapshot the current cfg state in order to pass to the inference
    # subprocesses
    cfg_file = os.path.join(output_dir, '{}_range_config.yaml'.format(tag))
    with open(cfg_file, 'w') as f:
        envu.yaml_dump(cfg, stream=f)
    subprocess_env = os.environ.copy()
    processes = []
    subinds = np.array_split(range(total_range_size), cfg.NUM_GPUS)
    # Determine GPUs to use
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        gpu_inds = map(int, cuda_visible_devices.split(','))
        assert -1 not in gpu_inds, \
            'Hiding GPU indices using the \'-1\' index is not supported'
    else:
        gpu_inds = range(cfg.NUM_GPUS)
    # Run the binary in cfg.NUM_GPUS subprocesses
    for i, gpu_ind in enumerate(gpu_inds):
        start = subinds[i][0]
        end = subinds[i][-1] + 1
        subprocess_env['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)
        cmd = '{binary} --range {start} {end} --cfg {cfg_file} NUM_GPUS 1 {opts}'
        cmd = cmd.format(
            binary=shlex_quote(binary),
            start=int(start),
            end=int(end),
            cfg_file=shlex_quote(cfg_file),
            opts=' '.join([shlex_quote(opt) for opt in opts])
        )
        logger.info('{} range command {}: {}'.format(tag, i, cmd))
        if i == 0:
            subprocess_stdout = subprocess.PIPE
        else:
            filename = os.path.join(
                output_dir, '%s_range_%s_%s.stdout' % (tag, start, end)
            )
            subprocess_stdout = open(filename, 'w')  # NOQA (close below)
        p = subprocess.Popen(
            cmd,
            shell=True,
            env=subprocess_env,
            stdout=subprocess_stdout,
            stderr=subprocess.STDOUT,
            bufsize=1
        )
        processes.append((i, p, start, end, subprocess_stdout))
    # Log output from inference processes and collate their results
    outputs = []
    for i, p, start, end, subprocess_stdout in processes:
        log_subprocess_output(i, p, output_dir, tag, start, end)
        if i > 0:
            subprocess_stdout.close()
        range_file = os.path.join(
            output_dir, '%s_range_%s_%s.pkl' % (tag, start, end)
        )
        range_data = load_object(range_file)
        outputs.append(range_data)
    return outputs


def log_subprocess_output(i, p, output_dir, tag, start, end):
    """Capture the output of each subprocess and log it in the parent process.
    The first subprocess's output is logged in realtime. The output from the
    other subprocesses is buffered and then printed all at once (in order) when
    subprocesses finish.
    """
    outfile = os.path.join(
        output_dir, '%s_range_%s_%s.stdout' % (tag, start, end)
    )
    logger.info('# ' + '-' * 76 + ' #')
    logger.info(
        'stdout of subprocess %s with range [%s, %s]' % (i, start + 1, end)
    )
    logger.info('# ' + '-' * 76 + ' #')
    if i == 0:
        # Stream the piped stdout from the first subprocess in realtime
        with open(outfile, 'wb') as f:
            for line in iter(p.stdout.readline, b''):
                print(line.rstrip().decode("utf8"))
                f.write(line)
        p.stdout.close()
        ret = p.wait()
    else:
        # For subprocesses >= 1, wait and dump their log file
        ret = p.wait()
        with open(outfile, 'r') as f:
            print(''.join(f.readlines()))
    assert ret == 0, 'Range subprocess failed (exit code: {})'.format(ret)
