
import torch
import torch.utils.data as data
import numpy as np
import pycocotools.coco as coco
import os
import cv2
import copy
import math

from ..utils.image import gaussian_radius, draw_umich_gaussian, get_affine_transform, affine_transform

class GenericDataset(data.Dataset):
  class_name = None
  cls_ids = None
  class_weights = None
  crop_range = [0,0,0,0]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  max_objs = 256


  def __init__(self, opt=None, split=None, ann_path=None, img_path=None):
    self.opt = opt
    self.split = split
    self.data_rng = np.random.RandomState(123)
    self.ann_path = ann_path
    self.img_path = img_path
    self.num_samples = 0

    if ann_path is not None and img_path is not None:
      print("==> initialize {} data from annotation {}, \n images from {}".format(
            split, ann_path, img_path))
      self.coco = coco.COCO(ann_path)
      self.imgIds = self.coco.getImgIds()
      self.num_samples = len(self.imgIds)
      self.img_path = img_path

  def __getitem__(self, index):
    #override required by DataLoader
    img, anns, img_info, img_path, depth_map = self._load_data(index)
    height, width = img.shape[0], img.shape[1]

    # Get tranform from raw input image to fixed-size input image, and
    # tranform from raw input image to fixed-size output image
    center = np.array([width/2., height/2.], dtype=np.float32)
    scale = max(img.shape[0], img.shape[1]) * 1.0
    if self.split == 'train':
      #TODO(): img + ann data flip/crop/truncation augument
      #TODO(): img color brightness/contrast/saturation augument
      trans_input = get_affine_transform(center, scale, 0, [self.opt.input_w, self.opt.input_h])
      trans_output = get_affine_transform(center, scale, 0, [self.opt.output_w, self.opt.output_h])
    
    inp = cv2.warpAffine(img, trans_input,
                       (self.opt.input_w, self.opt.input_h),
                       flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1) #[c, h, w]
    ret = {'image': inp}

    gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'center': [], 'xs': [], 'ys': [],
              'tracking_id': [], 'tl_bboxes': [], 'tl_scores': [], 'tl_clses': [],
              'tl_cts': [], 'aug_s': []}
    self._init_ret(ret, gt_det)
    calib, distort = self._get_calib(img_info, width, height)
    if self.opt.cls_reweight:
      ret['class_weight'] = self.weights_dict

    anns, bboxes_info = self._sort_bbox(anns, height, width, trans_output)
    num_bbox = len(bboxes_info)
    for i in range(num_bbox):
      bbox, bbox_amodal, ann = bboxes_info[i][0], bboxes_info[i][1], bboxes_info[i][2]
      cls_id = int(self.cls_ids[ann['category_id']])
      if cls_id <=0 or cls_id > self.opt.num_classes:
        continue
      self._add_instance(ret,gt_det,i,cls_id,bbox,bbox_amodal,ann,calib,trans_output)  
    return ret

  def _get_calib(self, img_info, width, height):
    if 'calib' in img_info:
      calib = np.array(img_info['calib'],dtype=np.float32)
    else:
      calib = np.zeros((3,4))

    if 'distort' in img_info:
      distort = np.array(img_info['distort'],dtype=np.float32)
    else:
      distort = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    return calib, distort

  def _init_ret_fpn(self, ret, gt_det):
    gt_det['stride'] = []
    ret['hm'] = {stride: np.zeros((self.opt.num_classes, self.opt.input_h // stride,
                 self.opt.input_w // stride), np.float32) for stride in self.opt.out_strides}
    ret['gt_center'] = np.zeros((self.max_objs, 2), dtype=np.float32)
    ret['gt_bboxes'] = np.zeros((self.max_objs, 4), dtype=np.float32)
    ret['gt_radius'] = np.zeros((self.max_objs, 2), dtype=np.int32)
    ret['gt_cls'] = np.zeros((self.max_objs,), dtype=np.int32) - 1

    ret['class_weight'] = {stride: np.ones(self.opt.num_classes, dtype=np.float32) \
                           for stride in self.opt.out_strides}
    ret['ind'] = {stride: np.zeros((self.max_objs), dtype=np.int64) \
                  for stride in self.opt.out_strides}
    ret['cat'] = {stride: np.zeros((self.max_objs), dtype=np.int64) \
                  for stride in self.opt.out_strides}
    ret['mask'] = {stride: np.zeros((self.max_objs), dtype=np.float32) \
                  for stride in self.opt.out_strides}
    ret['reg_class_weight'] = {stride: np.zeros((self.max_objs), dtype=np.float32) \
                               for stride in self.opt.out_strides}

    regression_head_dims = {
      'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
      'nuscenes_att': 8, 'velocity': 3, 'dep': self.opt.heads['dep'],
      'dim': 3, 'amodel_offset': 2, 'dep_off': 1,
      'pos_off': 3, 'rel_vel': 3, 'latd': 1, 'obs_dir': 4}
    for head in regression_head_dims:
      if head in self.opt.heads:
        ret[head] = {stride: np.zeros((self.max_objs, regression_head_dims[head]), dtype=np.float32) \
                     for stride in self.opt.out_strides}
        ret[head+'_mask'] = {stride: np.zeros((self.max_objs, regression_head_dims[head]), dtype=np.float32) \
                             for stride in self.opt.out_strides}
        gt_det[head] = []

    if 'rot' in self.opt.heads:
      ret['rotbin'] = {stride: np.zeros((self.max_objs, 2), dtype=np.int64) \
                       for stride in self.opt.out_strides}
      ret['rotres'] = {stride: np.zeros((self.max_objs, 2), dtype=np.float32) \
                       for stride in self.opt.out_strides}
      ret['rot_mask'] = {stride: np.zeros((self.max_objs), dtype=np.float32) \
                         for stride in self.opt.out_strides}
    gt_det.update({'rot': []})

  def _init_ret(self, ret, gt_det):
    gt_det['stride'] = []
    ret['hm'] = np.zeros((self.opt.num_classes, self.opt.input_h // 4,
                 self.opt.input_w // 4), np.float32)
    ret['gt_center'] = np.zeros((self.max_objs, 2), dtype=np.float32)
    ret['gt_bboxes'] = np.zeros((self.max_objs, 4), dtype=np.float32)
    ret['gt_radius'] = np.zeros((self.max_objs, 2), dtype=np.int32)
    ret['gt_cls'] = np.zeros((self.max_objs,), dtype=np.int32) - 1

    ret['class_weight'] = np.ones(self.opt.num_classes, dtype=np.float32)
    ret['ind'] = np.zeros((self.max_objs), dtype=np.int64)
    ret['cat'] = np.zeros((self.max_objs), dtype=np.int64)
    ret['mask'] = np.zeros((self.max_objs), dtype=np.float32)
    ret['reg_class_weight'] = np.zeros((self.max_objs), dtype=np.float32)

    regression_head_dims = {
      'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
      'nuscenes_att': 8, 'velocity': 3, 'dep': self.opt.heads['dep'],
      'dim': 3, 'amodel_offset': 2, 'dep_off': 1,
      'pos_off': 3, 'rel_vel': 3, 'latd': 1, 'obs_dir': 4}
    for head in regression_head_dims:
      if head in self.opt.heads:
        ret[head] = np.zeros((self.max_objs, regression_head_dims[head]), dtype=np.float32)
        ret[head+'_mask'] = np.zeros((self.max_objs, regression_head_dims[head]), dtype=np.float32)
        gt_det[head] = []

    if 'rot' in self.opt.heads:
      ret['rotbin'] = np.zeros((self.max_objs, 2), dtype=np.int64)
      ret['rotres'] = np.zeros((self.max_objs, 2), dtype=np.float32)
      ret['rot_mask'] = np.zeros((self.max_objs), dtype=np.float32)
      gt_det.update({'rot': []})

  def _add_instance(self, ret, gt_det, i, cls_id,bbox, bbox_amodal, ann, \
                    calib, trans_output=None):
    #bbox = [x1, y1, x2, y2]
    h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
    center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2], dtype=np.float32)
    center_int = np.round(center).astype(np.int32)  #[centerW, centerH]
    radius = gaussian_radius((math.ceil(h),math.ceil(w)))  #unit:pixel int
    radius = max(0, int(math.ceil(radius)))

    ret['gt_center'][i,:] = center
    ret['gt_cls'][i] = cls_id-1
    ret['gt_radius'][i,0] = radius
    ret['gt_radius'][i,1] = radius
    ret['gt_bboxes'][i,:] = bbox
    ret['cat'][i] = cls_id - 1
    ret['mask'][i] = 1
    ret['ind'][i] = center_int[1] * self.opt.output_w + center_int[0]
    if 'wh' in ret:
      ret['wh'][i] = [w, h]
      ret['wh_mask'][i] = 1
    if 'reg' in ret:
      ret['reg'][i] = center - center_int
      ret['reg_mask'][i] = 1
    #draw gaussian distribution on heatmap's specific category channel
    draw_umich_gaussian(ret['hm'][cls_id-1], center_int, radius)
    #[x1,y1,x2,y2]
    gt_det['bboxes'].append(np.array([center[0]-w/2, center[1]-h/2,
                                      center[0]+w/2, center[1]+h/2],dtype=np.float32))
    gt_det['scores'].append(1.)
    gt_det['clses'].append(cls_id-1)
    gt_det['center'].append(center)

    ret['reg_class_weight'][i] = 1.

    if 'rot' in self.opt.heads:
      self._add_rot(ret, ann, i, gt_det)

    if 'dep' in self.opt.heads:
      if 'depth' in ann:
        if ann['depth']>-5 and ann['ign']<0.5:
          ret['dep_mask'][i] = 1
          ret['dep'][i] = ann['depth']
          gt_det['dep'].append(ret['dep'][i])
        else:
          gt_det['dep'].append(2)
      else:
         gt_det['dep'].append(2)

    if 'dim' in self.opt.heads:
      if 'dim' in ann:
        if ann['depth']>-5 and ann['ign']<0.5:
          ret['dim_mask'][i] = 1
          ret['dim'][i] = ann['dim']
          gt_det['dim'].append(ret['dim'][i])
        else:
          gt_det['dim'].append([1,1,1])
      else:
        gt_det['dim'].append([1,1,1])

    if 'amodel_offset' in self.opt.heads:
      if 'amodel_center' in ann:
        if ann['depth']>-5 and ann['ign']<0.5:
          amodel_center = affine_transform(ann['amodel_center'], trans_output)
          ret['amodel_offset_mask'][i] = 1
          ret['amodel_offset'][i] = amodel_center - center_int
          gt_det['amodel_offset'].append(ret['amodel_offset'][i])
        else:
          gt_det['amodel_offset'].append([0,0])
      else:
        gt_det['amodel_offset'].append([0,0])

  def _add_rot(self, ret, ann, i, gt_det):
    if 'alpha' in ann:
      if ann['depth']>-5 and ann['ign']<0.5:
        ret['rot_mask'][i] = 1
        alpha = ann['alpha']
        if alpha < np.pi/6. or alpha > 5*np.pi/6.:
          ret['rotbin'][i,0] = 1
          ret['rotres'][i,0] = alpha - (-0.5*np.pi)
        if alpha > -np.pi/6. or alpha < -5*np.pi/6.:
          ret['rotbin'][i,1] = 1
          ret['rotres'][i,1] = alpha - (-0.5*np.pi)
        gt_det['rot'].append(self._alpha_to_8(ann['alpha']))
      else:
        gt_det['rot'].append(self._alpha_to_8(0))
    else:
      gt_det['rot'].append(self._alpha_to_8(0))

  def _alpha_to_8(self, alpha):
    res = [0,0,0,1,0,0,0,1]
    if alpha<np.pi/6. or alpha>5*np.pi/6.:
      r = alpha - (-0.5*np.pi)
      res[1] = 1
      res[2] = np.sin(r)
      res[3] = np.cos(r)
    if alpha>-np.pi/6. or alpha<-5*np.pi/6.:
      r = alpha - (0.5*np.pi)
      res[5] = 1
      res[6] = np.sin(r)
      res[7] = np.cos(r)
    return res


  def _sort_bbox(self, anns, height, width, trans_output=None):
    #sort all annotation bboxes of this image
    bboxes_info = []
    num_objs = len(anns)
    for i in range(num_objs):
      ann = anns[i]
      bbox, bbox_amodal = self._get_bbox_output(ann['bbox'], height, width, trans_output)

      bbox_info = []
      bbox_info.append(bbox)  #raw bbox
      bbox_info.append(bbox_amodal)  #after transformed bbox
      bbox_info.append(ann)
      bbox_info.append((bbox[2]-bbox[0])*(bbox[3]-bbox[1])) #raw bbox area
      bboxes_info.append(bbox_info)
    def sort_rule(elem):
      return elem[3]
    bboxes_info.sort(key=sort_rule, reverse=True)  #sort bbox based on increasing area
    return anns, bboxes_info


  def _get_bbox_output(self, bbox, height, width, trans_output=None):
    bbox = self.__coco_box_to_box(bbox).copy()
    rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], 
                     [bbox[2], bbox[3]], [bbox[2], bbox[1]]], dtype=np.float32)
    for t in range(4):
      rect[t] = affine_transform(rect[t], trans_output)  #transform rect bbox

    #get new bbox after rect transformed
    bbox[:2] = rect[:,0].min(), rect[:,1].min()
    bbox[2:] = rect[:,0].max(), rect[:,1].max()

    bbox_amodal = copy.deepcopy(bbox)
    #clip by image bound
    bbox[[0,2]] = np.clip(bbox[[0,2]],0,self.opt.output_w-1)
    bbox[[1,3]] = np.clip(bbox[[1,3]],0,self.opt.output_h-1)
    
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    return bbox, bbox_amodal



  def _get_aug_param(self, c, s, width, height, anns):
    c = copy.deepcopy(c)
    s = copy.deepcopy(s)
    aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))  #select one from 0.6~1.4
    w_border = self._get_border(128, width) #128
    h_border = self._get_border(128, height) #128
    c[0] = np.random.randint(low=w_border, high=width-w_border)
    c[1] = np.random.randint(low=h_border, high=height-h_border) #get a random center
    rot = 0 
    return c, aug_s, rot


  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
      i *= 2
    return border // i


  def __len__(self):
    return self.num_samples


  def _load_data(self, index):
    img_id = self.imgIds[index]
    #get infomation for this image
    img_info = self.coco.loadImgs([img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(self.img_path, file_name)
  
    #get all annotation boxes for this image
    annIds = self.coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(self.coco.loadAnns(annIds))  

    fake_map = self._get_fake_maps()  #all zero depth for each pixel
    img = cv2.imread(img_path)

    if False:
      cv2.imshow('image window', img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

    anns = self._anns_preprocess(anns)
    #_img_pre_crop(..)  

    # pixel-wise depth map
    if 'gd_depth' in self.opt.heads:
      depth_map_path = img_path.replace('images', depth_map_version).replace('.jpg', '.png')
      if os.path.exists(depth_map_path):
        depth_map = np.expand_dims(cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED), axis=-1)
        #TODO():follow aug and crop operatoins on image
      else:
        depth_map = fake_map['fake_depth_map']
    else:
      depth_map = None

    #image color augument
    if np.random.uniform() <= 0.0:
      img = self._img_color_aug(img, anns)

    return img, anns, img_info, img_path, depth_map


  def _img_pre_crop(self, img, anns, img_info):
    crop_range = [0,0,0,0]
    try:
      img_crop = img[crop_range[0] : self.opt.input_h-crop_range[2], \
                     crop_range[1] : self.opt.input_w-crop_range[3], :]
    except TypeError:
      print(img_info['file_name'], "crop failed")
      raise TypeError
    
    img = copy.deepcopy(img_crop)

    #TODO(): modify anns based on img crop






  def _img_color_aug(self, img, anns):
    pass
    ##convert some bbox color
    # for ann in anns:
    #   bbox = ann['bbox']
    #   img_crop = img[..box..]
    #   img_aug = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    #   img_aug = cv2.cvtColor(img_aug, cv2.COLOR_GRAY2BGR)
    #   img_crop = _img_aug
    # return img



  def _anns_preprocess(self, anns):
    #preprocess 3d information
    if self.opt.obj_depth_loss_type == 'L1' and \
       self.opt.obj_depth_encode_type == 'ori':
      #for each annotation box
      for ann in anns:
        if 'depth' in ann:
          ann['ign'] = 0.
        if 'alpha' in ann:
          rot_y = ann['rotation_y']
          loc = ann['location']
          alpha_new = rot_y - np.arctan2(loc[0], loc[2])
          if alpha_new > np.pi:
            alpha_new -= 2*np.pi
          if alpha_new < -np.pi:
            alpha_new += 2*np.pi
          ann['alpha'] = alpha_new
        if 'amodel_center' in ann:
          ann['amodel_center'] = [-10.0, -10.0]
    return anns



  def _get_fake_maps(self):
    fake_depth_map = np.zeros((self.default_resolution[1],
                               self.default_resolution[0]), dtype=np.uint16)
    return {'fake_depth_map': fake_depth_map}

    
  def __coco_box_to_box(self, coco_box):
    #convert [x1,y1,w,h] to [x1,y1,x2,y2]
    bbox = np.array([coco_box[0], coco_box[1], coco_box[0]+coco_box[2],
                     coco_box[1]+coco_box[3]], dtype=np.float32)
    return bbox
