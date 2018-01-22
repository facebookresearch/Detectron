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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import tempfile
import unittest
import yaml

from core.config import cfg
from utils.collections import AttrDict
import core.config
import utils.logging


class TestCfg(unittest.TestCase):
    def test_copy_cfg(self):
        cfg2 = copy.deepcopy(cfg)
        s = cfg.MODEL.TYPE
        cfg2.MODEL.TYPE = 'dummy'
        assert cfg.MODEL.TYPE == s

    def test_merge_cfg_from_cfg(self):
        # Test: merge from deepcopy
        s = 'dummy0'
        cfg2 = copy.deepcopy(cfg)
        cfg2.MODEL.TYPE = s
        core.config.merge_cfg_from_cfg(cfg2)
        assert cfg.MODEL.TYPE == s

        # Test: merge from yaml
        s = 'dummy1'
        cfg2 = yaml.load(yaml.dump(cfg))
        cfg2.MODEL.TYPE = s
        core.config.merge_cfg_from_cfg(cfg2)
        assert cfg.MODEL.TYPE == s

        # Test: merge with a valid key
        s = 'dummy2'
        cfg2 = AttrDict()
        cfg2.MODEL = AttrDict()
        cfg2.MODEL.TYPE = s
        core.config.merge_cfg_from_cfg(cfg2)
        assert cfg.MODEL.TYPE == s

        # Test: merge with an invalid key
        s = 'dummy3'
        cfg2 = AttrDict()
        cfg2.FOO = AttrDict()
        cfg2.FOO.BAR = s
        with self.assertRaises(KeyError):
            core.config.merge_cfg_from_cfg(cfg2)

        # Test: merge with converted type
        cfg2 = AttrDict()
        cfg2.TRAIN = AttrDict()
        cfg2.TRAIN.SCALES = [1]
        core.config.merge_cfg_from_cfg(cfg2)
        assert type(cfg.TRAIN.SCALES) is tuple
        assert cfg.TRAIN.SCALES[0] == 1

        # Test: merge with invalid type
        cfg2 = AttrDict()
        cfg2.TRAIN = AttrDict()
        cfg2.TRAIN.SCALES = 1
        with self.assertRaises(ValueError):
            core.config.merge_cfg_from_cfg(cfg2)

    def test_merge_cfg_from_file(self):
        with tempfile.NamedTemporaryFile() as f:
            yaml.dump(cfg, f)
            s = cfg.MODEL.TYPE
            cfg.MODEL.TYPE = 'dummy'
            assert cfg.MODEL.TYPE != s
            core.config.merge_cfg_from_file(f.name)
            assert cfg.MODEL.TYPE == s

    def test_merge_cfg_from_list(self):
        opts = [
            'TRAIN.SCALES', '(100, )', 'MODEL.TYPE', u'foobar', 'NUM_GPUS', 2
        ]
        assert len(cfg.TRAIN.SCALES) > 0
        assert cfg.TRAIN.SCALES[0] != 100
        assert cfg.MODEL.TYPE != 'foobar'
        assert cfg.NUM_GPUS != 2
        core.config.merge_cfg_from_list(opts)
        assert type(cfg.TRAIN.SCALES) is tuple
        assert len(cfg.TRAIN.SCALES) == 1
        assert cfg.TRAIN.SCALES[0] == 100
        assert cfg.MODEL.TYPE == 'foobar'
        assert cfg.NUM_GPUS == 2

    def test_deprecated_key_from_list(self):
        # You should see logger messages like:
        #   "Deprecated config key (ignoring): MODEL.DILATION"
        opts = ['FINAL_MSG', 'foobar', 'MODEL.DILATION', 2]
        with self.assertRaises(AttributeError):
            _ = cfg.FINAL_MSG  # noqa
        with self.assertRaises(AttributeError):
            _ = cfg.MODEL.DILATION  # noqa
        core.config.merge_cfg_from_list(opts)
        with self.assertRaises(AttributeError):
            _ = cfg.FINAL_MSG  # noqa
        with self.assertRaises(AttributeError):
            _ = cfg.MODEL.DILATION  # noqa

    def test_deprecated_key_from_file(self):
        # You should see logger messages like:
        #   "Deprecated config key (ignoring): MODEL.DILATION"
        with tempfile.NamedTemporaryFile() as f:
            cfg2 = copy.deepcopy(cfg)
            cfg2.MODEL.DILATION = 2
            yaml.dump(cfg2, f)
            with self.assertRaises(AttributeError):
                _ = cfg.MODEL.DILATION  # noqa
            core.config.merge_cfg_from_file(f.name)
            with self.assertRaises(AttributeError):
                _ = cfg.MODEL.DILATION  # noqa

    def test_renamed_key_from_list(self):
        # You should see logger messages like:
        #  "Key EXAMPLE.RENAMED.KEY was renamed to EXAMPLE.KEY;
        #  please update your config"
        opts = ['EXAMPLE.RENAMED.KEY', 'foobar']
        with self.assertRaises(AttributeError):
            _ = cfg.EXAMPLE.RENAMED.KEY  # noqa
        with self.assertRaises(KeyError):
            core.config.merge_cfg_from_list(opts)

    def test_renamed_key_from_file(self):
        # You should see logger messages like:
        #  "Key EXAMPLE.RENAMED.KEY was renamed to EXAMPLE.KEY;
        #  please update your config"
        with tempfile.NamedTemporaryFile() as f:
            cfg2 = copy.deepcopy(cfg)
            cfg2.EXAMPLE = AttrDict()
            cfg2.EXAMPLE.RENAMED = AttrDict()
            cfg2.EXAMPLE.RENAMED.KEY = 'foobar'
            yaml.dump(cfg2, f)
            with self.assertRaises(AttributeError):
                _ = cfg.EXAMPLE.RENAMED.KEY  # noqa
            with self.assertRaises(KeyError):
                core.config.merge_cfg_from_file(f.name)


if __name__ == '__main__':
    utils.logging.setup_logging(__name__)
    unittest.main()
