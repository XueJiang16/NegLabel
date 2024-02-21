# Negative Label Guided OOD Detection with Pretrained Vision-Language Models


Official PyTorch implementation of the ICLR 2024 paper:

**[Negative Label Guided OOD Detection with Pretrained Vision-Language Models](https://openreview.net/forum?id=xUO1HXz4an)**

Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, Bo Han

Abstract: *Out-of-distribution (OOD) detection aims at identifying samples from unknown classes, playing a crucial role in trustworthy models against errors on unexpected inputs. There is extensive research dedicated to exploring OOD detection in the vision modality. Vision-language models (VLMs) can leverage both textual and visual information for various multi-modal applications, whereas few OOD detection methods take into account information from the text modality. In this paper, we propose a novel post hoc OOD detection method, called NegLabel, which takes a vast number of negative labels from extensive corpus databases. We design a novel scheme for the OOD score collaborated with negative labels. Theoretical analysis helps to understand the mechanism of negative labels. Extensive experiments demonstrate that our method NegLabel achieves state-of-the-art performance on various OOD detection benchmarks and generalizes well on multiple VLM architectures. Furthermore, our method NegLabel exhibits remarkable robustness against diverse domain shifts.*

## Installation

The project is based on MMClassification. MMClassification is an open source image classification toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -e .
```

Then install more packages:

```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Dataset Preparation

#### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data (not necessary) and validation data in
`./data/id_data/imagenet_train` and  `./data/id_data/imagenet_val`, respectively.

#### Out-of-distribution dataset

Following [MOS](https://arxiv.org/pdf/2105.01879.pdf), we use the following 4 OOD datasets for evaluation: [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), [SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), [Places](http://places2.csail.mit.edu/PAMI_places.pdf), and [Textures](https://arxiv.org/pdf/1311.3618.pdf).

Please refer to [MOS](https://github.com/deeplearning-wisc/large_scale_ood), download OOD datasets and put them into `./data/ood_data/`.


## OOD Detection Evaluation

To reproduce our results, please run:

```bash
bash ./run.sh
```



## Citation


```
@inproceedings{jiang2024neglabel,
  title={Negative Label Guided OOD Detection with Pretrained Vision-Language Models},
author={Xue Jiang and Feng Liu and Zhen Fang and Hong Chen and Tongliang Liu and Feng Zheng and Bo Han},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
