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

from detectron.core.config import cfg
from detectron.utils.collections import AttrDict
import detectron.core.config as core_config
import detectron.utils.logging as logging_utils


class TestAttrDict(unittest.TestCase):
    def test_immutability(self):
        # Top level immutable
        a = AttrDict()
        a.foo = 0
        a.immutable(True)
        with self.assertRaises(AttributeError):
            a.foo = 1
            a.bar = 1
        assert a.is_immutable()
        assert a.foo == 0
        a.immutable(False)
        assert not a.is_immutable()
        a.foo = 1
        assert a.foo == 1

        # Recursively immutable
        a.level1 = AttrDict()
        a.level1.foo = 0
        a.level1.level2 = AttrDict()
        a.level1.level2.foo = 0
        a.immutable(True)
        assert a.is_immutable()
        with self.assertRaises(AttributeError):
            a.level1.level2.foo = 1
            a.level1.bar = 1
        assert a.level1.level2.foo == 0

        # Serialize immutability state
        a.immutable(True)
        a2 = core_config.load_cfg(yaml.dump(a))
        assert a.is_immutable()
        assert a2.is_immutable()


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
        core_config.merge_cfg_from_cfg(cfg2)
        assert cfg.MODEL.TYPE == s

        # Test: merge from yaml
        s = 'dummy1'
        cfg2 = core_config.load_cfg(yaml.dump(cfg))
        cfg2.MODEL.TYPE = s
        core_config.merge_cfg_from_cfg(cfg2)
        assert cfg.MODEL.TYPE == s

        # Test: merge with a valid key
        s = 'dummy2'
        cfg2 = AttrDict()
        cfg2.MODEL = AttrDict()
        cfg2.MODEL.TYPE = s
        core_config.merge_cfg_from_cfg(cfg2)
        assert cfg.MODEL.TYPE == s

        # Test: merge with an invalid key
        s = 'dummy3'
        cfg2 = AttrDict()
        cfg2.FOO = AttrDict()
        cfg2.FOO.BAR = s
        with self.assertRaises(KeyError):
            core_config.merge_cfg_from_cfg(cfg2)

        # Test: merge with converted type
        cfg2 = AttrDict()
        cfg2.TRAIN = AttrDict()
        cfg2.TRAIN.SCALES = [1]
        core_config.merge_cfg_from_cfg(cfg2)
        assert type(cfg.TRAIN.SCALES) is tuple
        assert cfg.TRAIN.SCALES[0] == 1

        # Test: merge with invalid type
        cfg2 = AttrDict()
        cfg2.TRAIN = AttrDict()
        cfg2.TRAIN.SCALES = 1
        with self.assertRaises(ValueError):
            core_config.merge_cfg_from_cfg(cfg2)

    def test_merge_cfg_from_file(self):
        with tempfile.NamedTemporaryFile() as f:
            yaml.dump(cfg, f)
            s = cfg.MODEL.TYPE
            cfg.MODEL.TYPE = 'dummy'
            assert cfg.MODEL.TYPE != s
            core_config.merge_cfg_from_file(f.name)
            assert cfg.MODEL.TYPE == s

    def test_merge_cfg_from_list(self):
        opts = [
            'TRAIN.SCALES', '(100, )', 'MODEL.TYPE', u'foobar', 'NUM_GPUS', 2
        ]
        assert len(cfg.TRAIN.SCALES) > 0
        assert cfg.TRAIN.SCALES[0] != 100
        assert cfg.MODEL.TYPE != 'foobar'
        assert cfg.NUM_GPUS != 2
        core_config.merge_cfg_from_list(opts)
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
        core_config.merge_cfg_from_list(opts)
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
            core_config.merge_cfg_from_file(f.name)
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
            core_config.merge_cfg_from_list(opts)

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
                core_config.merge_cfg_from_file(f.name)


if __name__ == '__main__':
    logging_utils.setup_logging(__name__)
    unittest.main()
