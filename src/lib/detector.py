import cv2
import copy
import numpy as np
import torch
from .model.model import create_model, load_model
from .dataset.dataset_factory import get_dataset
from .utils.image import gaussian_radius, draw_umich_gaussian, get_affine_transform, affine_transform
from .utils.utils import transpose_and_gather_feat, sigmoid
from .utils.decode import generic_decode
from .utils.post_process import generic_post_process

class Detector(object):
  def __init__(self, opts):
    self.opts = opts

    self.model = create_model(opts.arch, opts.heads, opts.head_conv, opts)
    self.model,_,_,_ = load_model(self.model, opts)
    self.model = self.model.to(opts.device)
    self.model.eval()  #eval model

    self.trained_dataset = get_dataset(opts.dataset)
    self.mean = np.array(self.trained_dataset.mean, dtype=np.float32).reshape(1,1,3)
    self.std = np.array(self.trained_dataset.std, dtype=np.float32).reshape(1,1,3)

  def run(self, img_path, input_meta={}):
    img = cv2.imread(img_path)
    pre_processed = False

    detections = []

    #do inputs preprocess
    img, meta = self.pre_process(img, input_meta)
    img = img.to(self.opts.device, non_blocking=False)
    #run model result
    outputs, dets = self.process(img)
    #do output postprocess
    dets = self.post_process(dets, meta)
    results = self.merge_results(dets)  #filter by confidence threshold
    return {'results':results}


  def post_process(self, dets, meta, scale=1):
    dets = generic_post_process(self.opts, dets, [meta['center']], [meta['scale']],
                                [meta['out_height']], [meta['out_width']], self.opts.num_classes,
                                [meta['calib']], [meta['height']], [meta['width']])
    return dets['ret']



  def process(self, img, pre_img=None, pre_inds=None, cam_mask=None,
              coord_inp=None):
      with torch.no_grad():
        #torch.cuda.synchronize()
        batch = {'image':img, 'pre_img':pre_img, 'pre_inds':pre_inds,
                 'cam_mask':cam_mask, 'coord_inp':coord_inp}
        output = self.model(batch)

        output = self.sigmoid_output(output)
        #torch.cuda.synchronize()

        dets = generic_decode(output, K=self.opts.K, opt=self.opts)
        # torch.cuda.synchronize()

        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()

        return output, dets

  def merge_results(self, detections):
    results = []
    for i in range(len(detections)):
      if detections[i]['score'] > self.opts.out_thresh:
        results.append(detections[i])
    return results


  def sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = sigmoid(output['hm'])
    if 'dep' in output:
      if self.opts.obj_depth_loss_type == 'L1' and \
         self.opts.obj_depth_encode_type == 'ori':
         output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1. + \
                         self.opts.obj_min_depth - 0.1
    if 'depth_uncertain' in output:
      pass

    return output

  def pre_process(self, img, input_meta={}):
    h, w = img.shape[0:2]  #original raw img dim before transform
    #follow same processure as training
    center = np.array([w / 2., h / 2.], dtype=np.float32)
    scale = max(h, w) * 1.0

    trans_input = get_affine_transform(center, scale, 0, \
                                       [self.opts.input_w, self.opts.input_h])
    trans_output = get_affine_transform(center, scale, 0, \
                                        [self.opts.output_w, self.opts.output_h]) 
    inp = cv2.warpAffine(img, trans_input,
                         (self.opts.input_w, self.opts.input_h),
                         flags=cv2.INTER_LINEAR) 
    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1) #[c, h, w]
    inp = inp.reshape(1, 3, self.opts.input_h, self.opts.input_w) #[1,c,h,w]
    inp = torch.from_numpy(inp) #conver to pytorch tensor, since no dataloader wrapper

    meta = {'calib': np.array(input_meta['calib'], dtype=np.float32)}
    meta.update({'center':center, 'scale':scale, 'height':h, 'width':w, 
                 'inp_width':self.opts.input_w, 'inp_height':self.opts.input_h,
                 'out_width':self.opts.output_w, 'out_height':self.opts.output_h,
                 'trans_input':trans_input, 'trans_output':trans_output})
    return inp, meta
