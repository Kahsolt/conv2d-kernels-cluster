#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import get_model
from data import *
from utils import *


def pgd(model:Module, images:Tensor, labels:Tensor, eps:float=0.03, alpha:float=0.001, steps:int=40, element_wise:bool=True):
  images = images.clone().detach()
  labels = labels.clone().detach()

  adv_images = images.clone().detach()
  adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
  adv_images = torch.clamp(adv_images, min=0, max=1).detach()

  for _ in tqdm(range(steps)):
    adv_images.requires_grad = True
    outputs = model(normalize(adv_images))

    if element_wise:
      loss = F.cross_entropy(outputs, labels, reduction='none')
      grad = torch.autograd.grad(loss, adv_images, grad_outputs=loss)[0]
    else:
      loss = F.cross_entropy(outputs, labels)
      grad = torch.autograd.grad(loss, adv_images)[0]

    v_loss = loss.mean().item()
    #print('minimizing loss:', v_loss)
    if v_loss == 0.0: break

    with torch.no_grad():
      adv_images = adv_images.detach() + alpha * grad.sign()
      delta = torch.clamp(adv_images - images, min=-eps, max=eps)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

  # assure valid rgb pixel
  adv_images = (adv_images * 255).round().div(255.0)

  return adv_images


def do_test(model:Module, dataloader:DataLoader, atk:bool=False, show:bool=False) -> tuple:
  ''' Clean Accuracy, Remnant Accuracy, Attack Success Rate, Prediction Change Rate '''

  total, correct, rcorrect, changed = 0, 0, 0, 0
  attacked = 0

  model.eval()
  for X, Y in tqdm(dataloader):
    X = X.to(device)
    Y = Y.to(device)

    if atk:
      AX = pgd(model, X, Y, args.eps, args.alpha, args.steps)
      if show:
        with torch.no_grad():
          DX = AX - X
          Linf = DX.abs().max(dim=0)[0].mean()
          L2   = DX.square().sum(dim=0).sqrt().mean()
          print(f'Linf: {Linf}')
          print(f'L2: {L2}')

        imshow(X, AX)

    with torch.inference_mode():
      pred    = model(normalize(X)) .argmax(dim=-1)
      if atk:
        pred_AX = model(normalize(AX)).argmax(dim=-1)

    total    += len(pred)
    correct  += (pred    == Y   ).sum().item()               # clean correct
    if atk:
      rcorrect += (pred_AX == Y   ).sum().item()               # adversarial still correct
      changed  += (pred_AX != pred).sum().item()               # prediction changed under attack
      attacked += ((pred == Y) * (pred_AX != Y)).sum().item()  # clean correct but adversarial wrong

    if show:
      print('Y:', Y)
      print('pred:', pred)
      print('pred_AX:', pred_AX)
      print(f'total: {total}, correct: {correct}, rcorrect: {rcorrect}, changed: {changed}, attacked: {attacked}')

  return [
    correct  / total   if total   else 0,
    rcorrect / total   if total   else 0,
    changed  / total   if total   else 0,
    attacked / correct if correct else 0,
  ]


def test(args, model:nn.Module=None):
  ''' Model '''
  model = model or get_model(args.model)

  ''' Ckpt '''
  fp = LOG_PATH / f'{args.name}.pth'
  if fp.exists():
    print(f'>> loading weights from fp')
    ckpt = torch.load(fp, map_location=device)
    model.load_state_dict(ckpt)

  if args.ckpt:
    print(f'>> loading weights from {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)

  ''' Data '''
  dataloader = get_dataloader(args.batch_size, shuffle=args.shuffle)
  
  ''' Test '''
  acc, racc, asr, pcr = do_test(model, dataloader, atk=args.atk, show=args.show_atk)
  print(f'Clean Accuracy:         {acc:.2%}')
  if args.atk:
    print(f'Remnant Accuracy:       {racc:.2%}')
    print(f'Prediction Change Rate: {pcr:.2%}')
    print(f'Attack Success Rate:    {asr:.2%}')


if __name__ == '__main__':
  test(get_args())
