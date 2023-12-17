#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/28 

import torchvision.models as M

from utils import Module, device


def get_model(name:str) -> Module:
  model: Module = getattr(M, name)(pretrained=True)
  model = model.eval().to(device)
  return model
