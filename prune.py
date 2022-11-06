#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/11/06 

import os
from argparse import ArgumentParser
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
try:
  from sklearnex import patch_sklearn
  patch_sklearn()
except:
  print('cannot import sklearnex, if you have Intel CPU, do `pip install scikit-learn-intelex` for calculation acceleration')
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import MODELS, get_model
from util import frac_to_str

stats = {
  # 'layer_name': ((n_kernel_original, n_kernel_clustered), (L0, L1, L2, Linf))
}

def optimal_number_of_clusters(wcss, n_clust_min, n_clust_max, step=1):
  x1, y1 = n_clust_min, wcss[0]
  x2, y2 = n_clust_max, wcss[len(wcss)-1]

  distances = []
  for i in range(len(wcss)):
    x0 = i+n_clust_min
    y0 = wcss[i]
    numerator = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distances.append(numerator/denominator)

  return distances.index(max(distances)) * step + n_clust_min

@torch.no_grad()
def cluster_kernels(name:str, layer:nn.Conv2d, args):
  print(f'[cluster_kernels] {name}')

  weight = layer.weight                             # [C_o, C_i, K_h, K_w]
  C_out, C_in, K_h, K_w = weight.shape

  data = weight.flatten(start_dim=1).cpu().numpy()  # [C_o, C_i*K_h*K_w], vectorize

  # reduce by fixed ratio
  if args.method == 'fixed':
    n_clust = round(C_out * args.fixed)
  
  # reduce by inertia limit
  if args.method == 'inertia':
    for n_clust in range(C_out-1, C_out//2, -1):
      model = KMeans(n_clust)
      model.fit(data)
      inertia = model.inertia_
      del model
      print(f'   n_clust: {n_clust}, inertia: {inertia}')
      if inertia > args.inertia: break

    n_clust = n_clust + 1

  # reduce by optimal `wcss` metric
  if args.method == 'wcss':
    n_clust_min, n_clust_max = round(C_out * args.wcss_from), round(C_out * args.wcss_to)
    rng = n_clust_max - n_clust_min + 1
    step = max(1, round(np.log2(rng) - 3))
    
    wcss = []
    for n_clust in range(n_clust_min, n_clust_max, step):
      model = KMeans(n_clust)
      model.fit(data)
      print(f'   n_clust: {n_clust}, inertia: {model.inertia_}')
      wcss.append(model.inertia_)
      del model
    
    if args.show: plt.plot(wcss) ; plt.title('wcss') ; plt.show()

    n_clust = optimal_number_of_clusters(wcss, n_clust_min, n_clust_max)


  model = KMeans(n_clust)
  pred = model.fit_predict(data).tolist()
  centroids = model.cluster_centers_
  print(f'decision: n_clust={model.n_clusters}, inertia={model.inertia_}')

  if args.show:
    plt.subplot(211) ; plt.title('hist') ; plt.hist(pred, bins=n_clust)
    plt.subplot(212) ; plt.title('freq') ; plt.plot(sorted(Counter(pred).values(), reverse=True))
    plt.show()

    distmat = np.abs(centroids[None, :, :] - centroids[:, None, :]).mean(axis=-1)
    sns.heatmap(distmat)
    plt.title('dist matrix')
    plt.show()

  # update kernels
  kernel_prototypes = centroids.reshape([-1, C_in, K_h, K_w])   # de-vectorize
  weight_prototypes = torch.from_numpy(kernel_prototypes[pred])
  if 'repalce with centroids':
    new_weight = weight_prototypes
  if not 'step away from centroids':
    d_w = weight - weight_prototypes
    new_weight = weight + d_w * 0.1
  layer.weight = Parameter(new_weight, requires_grad=True)
  
  # calc states
  d_weight = new_weight - weight
  d_weight_abs = d_weight.abs()
  n_param = weight.numel()

  L0   = (d_weight != 0).sum() / n_param
  L1   = d_weight_abs.sum() / n_param
  L2   = d_weight.square().sum().sqrt() / n_param
  Linf = d_weight_abs.max()

  stats[name] = (
    (C_out, n_clust),
    (L0, L1, L2, Linf),
  )


def prune(args):
  print(f'>> loading pretrained {args.model}')
  model = get_model(args.model).eval()

  for name, mod in model.named_modules():
    if isinstance(mod, nn.Conv2d):
      cluster_kernels(name, mod, args)
      if args.only_first_layer: break

  fp = os.path.join(args.out_path, f'{args.name}.pth')
  print(f'>> saving weights to {fp}')
  torch.save(model.state_dict(), fp)

  fp = os.path.join(args.out_path, f'{args.name}.txt')
  print(f'>> saving stats to {fp}')
  with open(fp, 'w', encoding='utf-8') as fh:
    for k, v in stats.items():
      lines = [
        f'[{k}] {v[0][0]} => {v[0][1]}',
        f'   L0:   {v[1][0]}',
        f'   L1:   {v[1][1]}',
        f'   L2:   {v[1][2]}',
        f'   Linf: {v[1][3]}',
      ]
      log = '\n'.join(lines)
      fh.write(log) ; fh.write('\n')
      print(log)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS)
  parser.add_argument('--method', default='fixed', choices=['fixed', 'wcss', 'inertia'])
  parser.add_argument('--fixed',     default=0.9,  type=float, help='reduction ratio in range (0, 1)')
  parser.add_argument('--wcss_from', default=0.5,  type=float, help='reduction ratio in range (0, 1)')
  parser.add_argument('--wcss_to',   default=0.75, type=float, help='reduction ratio in range (0, 1)')
  parser.add_argument('--inertia',   default=2.0, type=float,  help='cluster inertia limit')

  parser.add_argument('--only_first_layer', action='store_true')
  
  parser.add_argument('--show', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--data_path', default='data')
  parser.add_argument('--out_path', default='out')
  args = parser.parse_args()

  os.makedirs(args.data_path, exist_ok=True)
  os.makedirs(args.out_path, exist_ok=True)

  if args.method == 'fixed':
    suffix = f'_fixed={frac_to_str(args.fixed)}'
  if args.method == 'wcss':
    suffix = f'_wcss={frac_to_str(args.wcss_from)}~{frac_to_str(args.wcss_to)}'
  if args.method == 'inertia':
    suffix = f'_inertia={frac_to_str(args.inertia)}'
  suffix += '_ofl' if args.only_first_layer else ''
  args.name = f'{args.model}{suffix}'
  
  fp = os.path.join(args.out_path, f'{args.name}.pth')
  if not args.overwrite and os.path.exists(fp):
    print(f'safely ignore {args.name} due to file exists: {fp}')
    exit(0)

  prune(args)
