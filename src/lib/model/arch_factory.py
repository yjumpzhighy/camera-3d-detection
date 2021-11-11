
from .backbones.mobilenet import MobileNetV2
from .necks.dlaup import DLASeg

backbone_factory = {'resnet':None, 'mobilenet':MobileNetV2}

neck_factory = {'dlaup': DLASeg}

def get_backbone(backbone):
  return backbone_factory[backbone]

def get_neck(neck):
  return neck_factory[neck]
