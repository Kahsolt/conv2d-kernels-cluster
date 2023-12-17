#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/11/06 

from collections import Counter

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
try:
  from sklearnex import patch_sklearn
  patch_sklearn()
except:
  print('>> cannot import sklearnex, if you have Intel CPU, do `pip install scikit-learn-intelex` for calculation acceleration')
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import *
from utils import *


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
def cluster_kernels(args:Namespace, name:str, layer:nn.Conv2d) -> Tuple:
  print(f'[cluster_kernels] {name}')

  weight = layer.weight                             # [C_o, C_i, K_h, K_w]
  C_out, C_in, K_h, K_w = weight.shape

  kernels = weight.flatten(start_dim=1).cpu().numpy()  # [C_o, C_i*K_h*K_w], vectorize

  # reduce by fixed ratio
  if args.method == 'fixed':
    n_clust = round(C_out * args.fixed)
  
  # reduce by inertia limit
  if args.method == 'inertia':
    for n_clust in range(C_out-1, C_out//2, -1):
      model = KMeans(n_clust, n_init='auto')
      model.fit(kernels)
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
      model = KMeans(n_clust, n_init='auto')
      model.fit(kernels)
      print(f'   n_clust: {n_clust}, inertia: {model.inertia_}')
      wcss.append(model.inertia_)
      del model
    
    if args.show_prune: plt.plot(wcss) ; plt.title('wcss') ; plt.show()

    n_clust = optimal_number_of_clusters(wcss, n_clust_min, n_clust_max)

  model = KMeans(n_clust, n_init='auto')
  pred = model.fit_predict(kernels).tolist()
  centroids = model.cluster_centers_    # [C_o_cls=58, 147=C_i*K_h*K_w]
  print(f'decision: n_clust={model.n_clusters}, inertia={model.inertia_}')

  if args.show_prune:
    plt.subplot(211) ; plt.title('hist') ; plt.hist(pred, bins=n_clust)
    plt.subplot(212) ; plt.title('freq') ; plt.plot(sorted(Counter(pred).values(), reverse=True))
    plt.show()

    sns.heatmap(np.abs(centroids[None, :, :] - centroids[:, None, :]).mean(axis=-1))
    plt.title('kernel L1-dist matrix')
    plt.show()

  # update kernels
  kernel_prototypes = centroids.reshape([-1, C_in, K_h, K_w]) # de-vectorize
  weight_prototypes = torch.from_numpy(kernel_prototypes[pred]).to(weight.device, weight.dtype)   # compress kernels
  if 'repalce with centroids':
    new_weight = weight_prototypes
  if not 'step away from centroids':    # experimental
    d_w = weight - weight_prototypes
    new_weight = weight + d_w * 0.1
  layer.weight = Parameter(new_weight, requires_grad=weight.requires_grad)

  # calc states
  d_weight = new_weight - weight
  d_weight_abs = d_weight.abs()
  n_param = weight.numel()

  L0   = (d_weight != 0).sum() / n_param
  L1   = d_weight_abs.sum() / n_param
  L2   = d_weight.square().sum().sqrt() / n_param
  Linf = d_weight_abs.max()

  return (
    (C_out, n_clust),
    (L0, L1, L2, Linf),
  )


def prune(args, model:nn.Module=None, log_fp:Path=LOG_PATH) -> nn.Module:
  if model is None:
    print(f'>> loading pretrained {args.model}')
    model = get_model(args.model)

  fp_cache = LOG_PATH / f'{args.name}.pth'
  if not fp_cache.exists():
    print(f'>> loaddng pruned weights from {fp_cache}')
    state_dict = torch.load(fp_cache)
    model.load_state_dict(state_dict)
    return model

  # 'layer_name': ((n_kernel_original, n_kernel_clustered), (L0, L1, L2, Linf))
  stats = {}
  for name, mod in model.named_modules():
    if isinstance(mod, nn.Conv2d):
      stats[name] = cluster_kernels(args, name, mod)
      if args.only_first_layer: break

  if log_fp:
    fp_stats = log_fp / f'{args.name}.txt'
    print(f'>> saving stats to {fp_stats}')
    with open(fp_stats, 'w', encoding='utf-8') as fh:
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

  if not fp_cache.exists():
    print(f'>> saving pruned weights to {fp_cache}')
    torch.save(model.state_dict(), fp_cache)

  return model


if __name__ == '__main__':
  args = get_args()
  model = prune(args)
