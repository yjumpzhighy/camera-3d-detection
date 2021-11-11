import numpy as np
import json
import os

from .kitti import KITTI
from .coco import COCO

dataset_factory = {
  'kitti': KITTI,
  'coco': COCO,
}

def get_dataset(dataset):
  return dataset_factory[dataset]
