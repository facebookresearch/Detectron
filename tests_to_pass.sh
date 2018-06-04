#!/bin/bash
set -ex
TDIR=detectron/tests

#python $TDIR/data_loader_benchmark.py
python $TDIR/test_batch_permutation_op.py
python $TDIR/test_bbox_transform.py
python $TDIR/test_cfg.py
python $TDIR/test_loader.py
#python $TDIR/test_restore_checkpoint.py
python $TDIR/test_smooth_l1_loss_op.py
python $TDIR/test_spatial_narrow_as_op.py
#python $TDIR/test_zero_even_op.py
