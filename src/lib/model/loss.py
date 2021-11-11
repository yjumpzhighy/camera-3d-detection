from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import transpose_and_gather_feat, sigmoid


def neg_loss(pred, gt, cls_weights=None):
  """
  Arguments:
      pred (batch x c x h x w)
      gt   (batch x c x h x w)
  """
  pos_inds = gt.eq(1).float()   #positive sample
  neg_inds = (gt.lt(1) & gt.gt(-0.5)).float()   #negative sample
  neg_weights = torch.pow(1-gt, 4)

  #pos_loss = -0.25 * torch.log(pred) * torch.pow(1-pred, 2) * pos_inds
  #neg_loss = -0.75 * torch.log(1-pred) * torch.pow(pred, 2) * neg_inds #* neg_weights
  pos_loss = torch.log(pred) * torch.pow(1-pred, 2) * pos_inds
  neg_loss = torch.log(1-pred) * torch.pow(pred, 2) * neg_inds * neg_weights
  
  if cls_weights is not None:
    cls_weights = cls_weights.unsqueeze(-1).unsqueeze(-1)
    pos_loss = pos_loss * cls_weights
    neg_loss = neg_loss * cls_weights

  num_pos = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  loss = 0
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def only_neg_loss(pred, gt):
  gt = torch.pow(1-gt, 4)
  neg_loss = torch.log(1-pred) * torch.pow(pred, 2) * gt
  return neg_loss.sum()


class FocalLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.neg_loss = neg_loss
  def forward(self, pred, gt, cls_weights=None):
    return self.neg_loss(pred, gt, cls_weights)

class RegL1Loss_dense(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, pred, reg_weight, gt):
    avg_factor = torch.sum(reg_weight>0).float().item() + 1e-6
    pred_shape = pred.shape
    wt_shape = reg_weight.shape
    reg_dim = pred_shape[1] // wt_shape[1]
    # (batch, 1*C, h, w) => (batch, dim*C, h, w)
    reg_weight = reg_weight.repeat(1, reg_dim, 1, 1) 
    # (batch, dim*C, h, w) => (batch, h, w, dim*C)
    reg_weight = reg_weight.permute(0,2,3,1)

    pred = pred.permute(0,2,3,1).contiguous()  #contiguous is just deepcopy
    gt = gt.permute(0,2,3,1).contiguous()

    pos_mask = reg_weight>0
    
class RegL1Loss(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, pred, mask, ind, tgt):
    x = transpose_and_gather_feat(pred, ind)
    loss = F.l1_loss(x*mask, tgt*mask, reduction='sum')
    loss = loss / (mask.sum()+1e-4)
    return loss


class BinRotLoss(nn.Module):
  def __init__(self):
    super().__init__()
  def _compute_rot_loss(self, pred, target_bin, target_res, mask):
    #to be implemented
    return 0.0


  def forward(self, pred, mask, ind, rotbin, rotres):
    x = transpose_and_gather_feat(pred, ind)
    loss = self._compute_rot_loss(x, rotbin, rotres, mask)
    return loss


class GenericLoss(nn.Module):
  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    self.crit = FocalLoss()
    self.crit_reg = RegL1Loss()
    if 'rot' in opt.heads:
      self.crit_rot = BinRotLoss()

  def forward(self, output, batch):
    device = batch['image'].device
    losses = {head: torch.sum(torch.zeros(1, device=device)) \
              for head in self.opt.heads}
    
    target = self._get_corr_stride_target(batch, 4)
    output = self._sigmoid_output(output)
    if 'hm' in output:
      #if self.opt.dense_encode:
      losses['hm'] += self.crit(output['hm'], target['hm'], target['class_weight'])
      print('hm loss:', losses['hm'])
    
    regression_heads = ['reg', 'wh', 'dep', 'dim', 'amodel_offset', 'latd']
    for head in regression_heads:
      if head in output:
        losses[head] += self.crit_reg(output[head], target[head+'_mask'], \
                        target['ind'], target[head])
        print(head+':', losses[head])
    
    if 'rot' in output:
      losses['rot'] += self.crit_rot(output['rot'], target['rot_mask'], \
                        target['ind'], target['rotbin'], target['rotres'])
      print('rot:', losses['rot'])

    losses['tot'] = 0
    for head in self.opt.heads:
      losses['tot'] += self.opt.weights_dict[head] * losses[head]    

    losses_gn = {}   #gradient nomralization
    return losses['tot'], losses, losses_gn
   

  def _get_corr_stride_target(self, batch, s):
    target = {}
    for key, val in batch.items():
      if type(val)==dict and key!='meta':
        target[key]=val[s]
      else:
        target[key]=val
    return target
      
  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = sigmoid(output['hm'])
    if 'dep' in output:
      if self.opt.obj_depth_loss_type == 'L1' and \
         self.opt.obj_depth_encode_type == 'ori':
         output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1. + \
                         self.opt.obj_min_depth - 0.1
    if 'depth_uncertain' in output:
      pass

    return output
