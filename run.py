#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/17 

from prune import prune
from test import test
from utils import get_args


if __name__ == '__main__':
  args = get_args()
  model = prune(args)
  test(args, model)
