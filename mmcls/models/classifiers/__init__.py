# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .vit_lt import VitClassifier
from .multi_modal import CLIPScalableClassifier

__all__ = ['BaseClassifier', 'ImageClassifier', 'VitClassifier', 'CLIPScalableClassifier']
