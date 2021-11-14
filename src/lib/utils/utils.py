import torch
import torch.nn as nn
import numpy as np


def gather_feat(feat, ind):
  """
  feat=[b,h*w,c]
  ind=[b,k]
  expand to [b,k,3] and get from [b,h*w,c]
  """
  dim = feat.size(2)  
  ind = ind.unsqueeze(2).expand(ind.size(0),ind.size(1),dim)
  feat = feat.gather(1, ind)
  return feat


def transpose_and_gather_feat(feat, ind):
  """
  feat=[b,w,h,c], transpose to [b,h,w,c] first, then
  get [b,k] from [b,h*w,c]
  """
  feat = feat.permute(0,2,3,1).contiguous()   #deep copy[b,h,w,c]
  feat = feat.view(feat.size(0), -1, feat.size(3)) #[b,h*w,c]
  feat = gather_feat(feat, ind)
  return feat


def sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def nms(heat, kernel=3, ch_pool=False):
  #find max scores value (after sigmoid) in 3*3 area
  pad = (kernel - 1) // 2
  hmax = nn.functional.max_pool2d(heat, (kernel,kernel),stride=1, padding=pad)
  keep = (hmax==heat).float()
  return heat*keep

def topK(scores, K):
  #topK on each channel first,then topK acrossing channels
  batch, cat, height, width = scores.size()
  #[b,c,h*w] -> [b,c,k]
  topk_scores, topk_indices = torch.topk(scores.view(batch,cat,-1), K)
  #topk_indices = topk_inds % (height * width)
  topk_ys = (topk_indices / width).int().float()
  topk_xs = (topk_indices % width).int().float()

  #[b,c,k] -> [b,k]
  topk_scores, topk_indice = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_indice/K).int() #get cls of each indice
  #get [b,k] from [b,ck,1]
  topk_indices = gather_feat(topk_indices.view(batch,-1,1), topk_indice).view(batch,K)
  topk_ys = gather_feat(topk_ys.view(batch,-1,1), topk_indice).view(batch,K)
  topk_xs = gather_feat(topk_xs.view(batch,-1,1), topk_indice).view(batch,K)
  return topk_scores, topk_indices, topk_clses, topk_ys, topk_xs


def get_multi_scale_topk_dets(dets, K):
  def sort_rule(elem):
    return elem['score']
  dets.sort(key=sort_rule, reverse=True)
  det_topk = dets[0:K]
  return det_topk


def get_clsreg_alpha(rot, num_bins, opt=None):
  """
  rot: (2*num_bins)
  """
  bins = rot[0:num_bins]
  reg = rot[num_bins:]
  ps = np.exp(bins)
  ps /= np.sum(ps)
  head_bin = np.argmax(bins)
  head_reg = reg[head_bin]
  alpha = bin2angle(head_bin, head_reg, num_bins)
  return alpha, ps, reg

def bin2angle(pred_bin, reg, num_bins):
  """
  invert function with angle2bin
  """
  angle_per_bin = 2 * np.pi / float(num_bins)
  angle_center = pred_bin * angle_per_bin
  angle = angle_center + reg  #rad
  if angle > np.pi:
    angle = angle - 2*np.pi
  return angle








class AverageMeter(object):
  def __init__(self):
    self.reset()
  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.cnt = 0
  def update(self, val, count=1):
    self.val = val
    self.sum += val*count
    self.cnt += count
    if self.cnt > 0:
      self.avg = self.sum / self.cnt

