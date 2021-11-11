import numpy as np
import torch
import os

from .generic_dataset import GenericDataset

class COCO(GenericDataset):
  default_resolution = [512, 512]
  num_categories = 80
  # class_name = ['SEDAN', 'TRUCK', 'PEDESTRIAN', 'BUS', 'BICYCLE', 'TRAFFIC_CONE_BUCKET',
  #               'ROAD_BARRIER', 'ANIMAL', 'EMERGENCYVEHICLE', 'TRAILER', 'VAN', 'TRICYCLE',
  #               'ROAD_DEBRIS', 'UNKNOWN_VEHICLE', 'OBJECT_UNKNOWN']
  class_name = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']
  num_categories = len(class_name)
  valid_ids = [i + 1 for i in range(num_categories)]
  cls_ids = {v: i + 1 for i, v in enumerate(valid_ids)}
  class_weights = np.array([1.0, 1.5, 5.0, 20.0, 1.0, 1.0, 1.0, 1.0, 5.0, \
                            15.0, 1.0, 1.0], dtype=np.float32)
  

  def __init__(self, opt, split):
    data_dir = os.path.join(opt.data_dir, 'coco')
    img_dir = os.path.join(data_dir, '{}2017'.format(split))
    ann_path = os.path.join(data_dir, 'annotations', 
                            'instances_{}2017.json').format(split)
    self.imgIds = None              
    super(COCO, self).__init__(opt, split, ann_path, img_dir)
    
    self.num_samples = len(self.imgIds)
    print('Loaded coco {} {} samples.'.format(split, self.num_samples))
   
