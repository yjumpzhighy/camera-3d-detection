import copy
import numpy as np

from .image import get_affine_transform, affine_transform, transform_preds_with_trans
from .ddd_utils import ddd2locrot
from .utils import get_clsreg_alpha, get_multi_scale_topk_dets

def generic_post_process(opts, det, center, scale, out_h, out_w, num_classes, calib,
                         h, w):

  """
  dets: each item include [1, K, x]
  center: [1,2]
  """
  for i in range(len(det['scores'])):  #for each batch
    preds = []
    #transform from output dimension to raw image dimension, to match raw image
    trans = get_affine_transform(center[i],scale[i],0,(out_w[i], out_h[i]),inv=1).astype(np.float32)
    for j in range(len(det['scores'][i])): #for each pred bbox
      item = {}
      item['score'] = det['scores'][i][j]
      item['class'] = int(det['clses'][i][j]) + 1
      item['centers'] = transform_preds_with_trans(det['centers'][i][j].reshape(1,2),
                                                   trans).reshape(2)
      if 'bboxes' in det:
        bbox = transform_preds_with_trans(det['bboxes'][i][j].reshape(2, 2), 
                                          trans).reshape(4)
        item['bbox'] = bbox #[4,]

      if 'dep' in det:
        item['dep'] = det['dep'][i][j]
      if 'latd' in det:
        out_latd = det['latd'][i][j]
        if opts.lateral_dst_reg_type == 'sqrt':
          item['latd'] = out_latd * np.abs(out_latd)
        else:
          raise ValueError('Unknown laterl dst reg type')
      if 'dim' in det:
        item['dim'] = det['dim'][i][j]
      if 'rot' in det:
        if opts.cls_reg_angle:
          item['alpha'], item['alpha_conf'], item['alpha_reg'] = \
            get_clsreg_alpha(det['rot'][i][j], opts.angle_cls_bin)
        else:
          raise ValueError('non rot regression not supported.')
      if 'rot' in det and 'dep' in det and 'dim' in det and item['dep']>opts.obj_min_depth:
        bbox = copy.deepcopy(item['bbox'])
        bbox = item['bbox']
        center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        if 'amodel_offset' in det:
          if opts.amodel_offset_reg_type == 'ori':
            decode_amodel_offset = det['amodel_offset'][i][j]
          elif opts.amodel_offset_reg_type == 'sqrt':
            decode_amodel_offset = det['amodel_offset'][i][j] * np.abs(det['amodel_offset'][i][j])
          else:
            raise ValueError('unknown amodel offset reg type.')
          
          if 'xs' in det:
            center_output = np.array([det['xs'][i][j], det['ys'][i][j]], dtype=np.float32)
          else:
            center_output = center

          amodel_center_output = center_output + decode_amodel_offset
          amodel_center_output = transform_preds_with_trans(amodel_center_output.reshape(1,2), 
                                                            trans).reshape(2).tolist()

          item['amodel_center'] = amodel_center_output
          item['loc'], item['rot_y'] = ddd2locrot(amodel_center_output, item['alpha'],
                                                  item['dim'],item['dep'],calib[i],
                                                  amodel_center_output)

      if 'dep_uncertainty' in det:
        if opts.dep_uncertainty_type == 'gaussian':
          item['dep_sigma'] = np.sqrt(np.exp(det['dep_uncertainty'][i][j]))
        else:
          raise ValueError('Unknow dep_uncertainty_type.')

      preds.append(item)

  #get topk dets (in case multiple strides and get more than K items)
  #then perform sort by scores
  ret_topk = get_multi_scale_topk_dets(preds, opts.K)
  return {'ret':ret_topk}
