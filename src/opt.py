


import argparse
import os
import sys


class Opts(object):
  def __init__(self):
    #self.parser = argparse.ArgumentParser()
    #task
    self.task = 'ddd'  #'ddd, lane'
    self.task = self.task.split(',')
    self.dataset = 'kitti' #'coco'
    self.test_dataset = 'kitti'  #'coco' 
    self.debug_mode = 0
    self.pretrained = False
    self.load_model_path = '' 
    self.save_model_path = ''
    self.lateral_dist = False 
    self.rel_dep_reg = False
    self.split = 'train' #'val'

    #system
    self.gpu_list = '' #'0'
    self.gpus = self.gpu_list.split(',') if self.gpu_list != '' else []
    self.dataloader_threads = 4
    self.use_cuda_dataloader = False
    self.seed = 36
    self.dist_train = False
    self.device = ''
    self.save_point = [1,20,40,60]
    self.root_dir = os.path.join(os.path.dirname(__file__), '..')
    self.save_dir = os.path.join(self.root_dir, 'checkpoints')
    
    #network
    self.arch = 'generic' #res_101, dla_34, mobilenet
    self.backbone = 'mobilenet'
    self.neck = 'dlaup'
    #['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']
    self.num_classes = 5 #12
    self.heads = {}
    self.num_head_conv = 1  #number of head conv layers
    self.head_conv = 24  #resnets:64,dla:256,mobilenet:24 channels num in each head conv layer 
    self.head_kernel = 3
    self.cls_reg_angle = True
    self.angle_cls_bin = 12  #bin number of angle
    self.upsample_node = 'ConvTranspose'
    #depth estimation
    self.obj_dep_fln = False #focal length normalization of object depth
    self.obj_dep_scln = False #scale normalization of object depth
    self.obj_dep_rotn = False #pitch normalization of object depth
    self.obj_depth_encode_type = 'ori'  #depth encode type
    self.obj_min_depth = 0.1
    self.obj_max_depth = 250.0
    self.dense_encode = False #???
    self.fpn = False
    self.out_strides = '4' #strides of output feature maps of fpn

    #data
    self.input_h = -1
    self.input_w = -1
    self.down_ratio = 4
    self.output_w = self.input_w // self.down_ratio
    self.output_h = self.input_h // self.down_ratio
    self.kitti_split = '3dop'  #'3dop | subcnn'
    self.use_coord_input = False
    self.data_channel = ['right-forward']  #which camera data source
    self.data_dir = '/home/zuyuan/Data'
    self.num_workers = 0 #dataloader threads.

    #train
    self.optimizer = 'Adam'
    self.lr = 0.0001
    self.lr_step = [60]  #drop learning rate by 10
    self.save_step = 90  #when to save the model to disk
    self.epochs = 60
    self.batch_size = 16
    
    #loss
    self.obj_depth_loss_type = 'L1' #'ord_reg'
    self.ord_num = 40 #'bin number of dense depth'
    self.dep_uncertain_type = 'gaussian'
    self.weights_dict = {'hm':1.0, 'wh':0.1, 'reg':1.0, 'dep':1.0, 
                         'rot':1.0, 'depth_uncertain':0.1, #'latd':1.0,
                         'dim':1.0, 'amodel_offset':1.0
                        }
    self.cls_reweight = False
    self.use_uncertainty_wt = False  #Automatic Weighted Loss
    self.use_grad_norm = False  #Gradient Normalize Loss
    self.use_modified_grad = False  #Modified gradient normalize
    self.use_dynamic_wt_avg = False  #Dynamic weights average loss
    self.crit_loss_type = 'Focal' #GHM-C:gradient harmonizing mechanism


  def update_with_dataset(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    

    opt.heads = {'hm': opt.num_classes, 'reg': 2, 'wh':2}
    if 'ddd' in opt.task:
      if opt.cls_reg_angle:
        rot_dim = opt.angle_cls_bin * 2  #24

      if opt.obj_depth_loss_type == 'ord_reg':
        dep_dim = opt.ord_num * 2 + 1
      else:
        dep_dim = 1

      opt.heads.update({'dep':dep_dim, 'rot':rot_dim, 'dim':3, 'amodel_offset':2})

      if (opt.dep_uncertain_type == 'gaussian'):
        opt.heads.update({'depth_uncertain':1})

      if opt.lateral_dist:
        opt.heads.update({'latd': 1})

      # update heads
      opt.weights = {head: self.weights_dict[head] for head in opt.heads}
      opt.head_conv = {head: [opt.head_conv for i in range(opt.num_head_conv if
                       head!='reg' else 1)] for head in opt.heads}
      opt.out_strides = [int(i) for i in opt.out_strides.split(',')]
      if not opt.fpn:
        opt.out_strides = [4]

    return opt
