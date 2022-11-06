#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

import os
import json
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


class ImageNet_1k(Dataset):

  def __init__(self, root: str):
    self.base_path = os.path.join(root, 'val')

    fns = [fn for fn in os.listdir(self.base_path)]
    fps = [os.path.join(self.base_path, fn) for fn in fns]
    with open(os.path.join(root, 'image_name_to_class_id_and_name.json'), encoding='utf-8') as fh:
      mapping = json.load(fh)
    tgts = [mapping[fn]['class_id'] for fn in fns]

    self.metadata = [x for x in zip(fps, tgts)]

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    fp, tgt = self.metadata[idx]
    img = Image.open(fp)
    img = img.convert('RGB')

    if 'use numpy':
      im = np.array(img, dtype=np.uint8).transpose(2, 0, 1)   # [C, H, W]
      im = im / np.float32(255.0)
    else:
      im = T.ToTensor()(img)

    return im, tgt


def normalize(X: torch.Tensor) -> torch.Tensor:
  ''' NOTE: to insure attack validity, normalization is delayed until put into model '''

  mean = (0.485, 0.456, 0.406)
  std  = (0.229, 0.224, 0.225)
  X = TF.normalize(X, mean, std)       # [B, C, H, W]

  return X


def imshow(X, AX, title=''):
  DX = X - AX
  DX = (DX - DX.min()) / (DX.max() - DX.min())

  grid_X  = make_grid( X).permute([1, 2, 0]).detach().cpu().numpy()
  grid_AX = make_grid(AX).permute([1, 2, 0]).detach().cpu().numpy()
  grid_DX = make_grid(DX).permute([1, 2, 0]).detach().cpu().numpy()
  plt.subplot(131) ; plt.title('X')  ; plt.axis('off') ; plt.imshow(grid_X)
  plt.subplot(132) ; plt.title('AX') ; plt.axis('off') ; plt.imshow(grid_AX)
  plt.subplot(133) ; plt.title('DX') ; plt.axis('off') ; plt.imshow(grid_DX)
  plt.tight_layout()
  plt.suptitle(title)

  try:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()    # 'QT4Agg' backend
  except: pass
  plt.show()


def get_dataloader(data_path, batch_size=100, shuffle=False):
  dataset = ImageNet_1k(root=os.path.join(data_path, 'imagenet-1k'))
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False, num_workers=0)
  return dataloader
