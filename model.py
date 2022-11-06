#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/28 

import warnings
warnings.simplefilter('ignore')

import torchvision.models as M

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
  'resnext50_32x4d',
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
#  'shufflenet_v2_x2_0',
]

def get_model(name):
  return getattr(M, name)(pretrained=True)
