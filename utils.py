#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import warnings ; warnings.filterwarnings('ignore', category=UserWarning)

import random
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import *

import torch
from torch import Tensor
from torch.nn import Module
import numpy as np

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)

RAND_SEED = 114514
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
  torch.cuda.manual_seed_all(RAND_SEED)
  torch.backends.cudnn.enabled = False
  torch.backends.cudnn.benchmark = False

MODELS = [
#  'googlenet',

#  'inception_v3',

#  'mobilenet_v2',
  'mobilenet_v3_small',
#  'mobilenet_v3_large',

  'resnet18',
#  'resnet34',
  'resnet50',
#  'resnet101',
#  'resnet152',
#  'resnext50_32x4d',
#  'resnext101_32x8d',
#  'resnext101_64x4d',
  'wide_resnet50_2',
#  'wide_resnet101_2',

  'densenet121',
#  'densenet161',
#  'densenet169',
#  'densenet201',

#  'shufflenet_v2_x0_5',
#  'shufflenet_v2_x1_0',
#  'shufflenet_v2_x1_5',
  'shufflenet_v2_x2_0',
]


def get_parser() -> ArgumentParser:
  parser = ArgumentParser()
  # model
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='model to test (eval/attack)')
  # prune
  parser.add_argument('--method', default='inertia', choices=['fixed', 'wcss', 'inertia'])
  parser.add_argument('--fixed',     default=0.9,  type=float, help='reduction ratio in range (0, 1)')
  parser.add_argument('--wcss_from', default=0.5,  type=float, help='reduction ratio in range (0, 1)')
  parser.add_argument('--wcss_to',   default=0.75, type=float, help='reduction ratio in range (0, 1)')
  parser.add_argument('--inertia',   default=1.0, type=float,  help='cluster inertia limit > 0')
  parser.add_argument('--only_first_layer', action='store_true')
  parser.add_argument('--show_prune', action='store_true', help='show kernel assignments')
  # test
  parser.add_argument('--ckpt', type=Path, help='path to ckpt file')
  parser.add_argument('-B', '--batch_size', type=int, default=64)
  parser.add_argument('--shuffle', action='store_true')
  parser.add_argument('--atk', action='store_true', help='enable PGD attack')
  parser.add_argument('--eps',   type=float, default=8/255)
  parser.add_argument('--alpha', type=float, default=1/255)
  parser.add_argument('--steps', type=int, default=10)
  parser.add_argument('--show_atk', action='store_true', help='show X-AX compare')

  return parser


def get_args(parser:ArgumentParser=None) -> Namespace:
  parser = parser or get_parser()
  args, _ = parser.parse_known_args()

  if args.method == 'fixed':
    suffix = f'_fixed={frac_to_str(args.fixed)}'
  if args.method == 'wcss':
    suffix = f'_wcss={frac_to_str(args.wcss_from)}~{frac_to_str(args.wcss_to)}'
  if args.method == 'inertia':
    suffix = f'_inertia={frac_to_str(args.inertia)}'
  suffix += '_ofl' if args.only_first_layer else ''
  args.name = f'{args.model}{suffix}'

  if args.ckpt:
    assert Path(args.ckpt).is_file()
    args.model = Path(args.ckpt).name.split('_')[0]

  return args


def frac_to_str(x:float, n_prec:int=2) -> str:
  x = round(x, n_prec)
  s = f'{x:f}'
  while s.endswith('0'): s = s[:-1]
  if s.endswith('.'): s += '0'
  return s
