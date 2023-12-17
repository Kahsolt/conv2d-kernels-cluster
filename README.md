# conv2d-kernels-cluster

    Try simplifying a CNN by conv-kernel clustering!

----

This is inspired by [https://github.com/Kahsolt/conv2d-kernels](https://github.com/Kahsolt/conv2d-kernels) and [https://github.com/Kahsolt/adversarial-prune](https://github.com/Kahsolt/adversarial-prune). Once we get to know that:

  - NNs are stable to small weights perturbation
  - there are many duplicated/redundant conv2d kernels in pretrained CNNs

Now consider: what if we replace all the kernels with its clustered centroids, will this network still work well?

结论:

- 用各种方法去合并核都会渐进地导致分类精度下降，是否可以在训练时引入一个正则损失以提高核的多样性？
- 核的多样性
  - 第一层 conv 会有大量相似冗余的核 (这个特性是否有助于将对抗样本问题杀死在第一层？)
  - 所有的 downsample 层都会出现大量相似冗余的核，因为 conv1x1 可能仅仅做了通道扩展
  - 其他层的核多样性都比较好，尤其是 layer4
- 核合并后更容易受对抗攻击了


### Experiments

⚪ ResNet18 on ImageNet-1k(*):

- Prune all layers

| Model | Accuracy | Note |
| :-: | :-: | :-: |
| pretrained | 91.80% | |
| fixed=0.99 | 90.50% | fix prune ratio |
| fixed=0.98 | 88.30% |  |
| fixed=0.95 | 85.30% |  |
| fixed=0.9  | 64.90% |  |
| fixed=0.85 | 33.10% |  |
| fixed=0.8  | 12.80% |  |
| fixed=0.75 |  3.30% |  |
| wcss=0.9~1.0  | 83.20% | auto prune ratio by wcss |
| wcss=0.75~1.0 | 34.70% |  |
| inertia=1 | **90.50%** | auto prune ratio by inertia |
| inertia=2 | 85.40% |  |
| inertia=3 | 83.50% |  |
| inertia=5 | 72.90% |  |

- Prune only first layer (`--only_first_layer`)

| Model | Accuracy | Note |
| :-: | :-: | :-: |
| pretrained | 91.80% | |
| fixed=0.95   | 91.80% | inertia:  4-14 |
| fixed=0.75   | 90.20% | inertia:  2.26 |
| fixed=0.5    | 53.20% | inertia: 19.18 |
| wcss=0.7~0.8 | 90.60% | inertia:  1.80 |
| wcss=0.4~0.6 | 51.00% | inertia: 23.82 |

(*) Test data is 1k samples drawn from validtion set of the whole ImageNet


### Quick Start

- cluster-based prune a pretrianed model: `python prune.py -M <model>`
- test original pretrained model: `python test.py -M <model>`
- test cluster-based pruned model: `python test.py --ckpt out\<model>_pruned.pth`

ℹ you can firstly run `run.cmd` for demo experiments

----

by Armit
2022/11/05 
