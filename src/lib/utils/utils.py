import torch

def gather_feat(feat, ind):
  """
  feat=[b,h*w,c]
  ind=[x,y]
  """
  dim = feat.size(2)  
  #[x,y]->[x,y,1]->[x,y,c]
  ind = ind.unsqueeze(2).expand(ind.size(0),ind.size(1),dim)
  feat = feat.gather(1, ind)
  return feat


def transpose_and_gather_feat(feat, ind):
  """
  feat=[b,w,h,c]
  """
  feat = feat.permute(0,2,3,1).contiguous()   #deep copy[b,h,w,c]
  feat = feat.view(feat.size(0), -1, feat.size(3)) #[b,h*w,c]
  feat = gather_feat(feat, ind)
  return feat


def sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y


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

