#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os.path as ops

def init_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_dir', type=str,
                      help='Where you store the dataset')
  parser.add_argument('--weights_path', type=str,
                      help='Where you store the pretrained weights')
  parser.add_argument('--vis', type=str, default=None)
  args = parser.parse_args()

  if not ops.exists(args.dataset_dir):
    raise ValueError('{:s} doesn\'t exist'.format(args.dataset_dir))

  return args
