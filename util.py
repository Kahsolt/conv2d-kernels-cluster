#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import os
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, 'data')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random

import numpy as np
import torch

RAND_SEED = 114514
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
  torch.cuda.manual_seed(RAND_SEED)
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True


def frac_to_str(x:float, n_prec:int=2) -> str:
  x = round(x, n_prec)
  s = f'{x:f}'
  while s.endswith('0'): s = s[:-1]
  if s.endswith('.'): s += '0'
  return s
