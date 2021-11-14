import torch

from .utils import nms, topK, transpose_and_gather_feat




def generic_decode(output, K=100, opt=None):
  if 'hm' not in output:
    return {}

  batch, cat, height, width = output['hm'].size()
  heat = nms(output['hm'])
  scores, indices, clses, ys0, xs0 = topK(heat, K=K)
  clses = clses.view(batch, K)
  scores = scores.view(batch, K)

  bboxes = None
  #get [b,K,2], indicate (x_w,y_h)
  centers = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)],dim=2)
  ret = {'scores':scores, 'clses':clses.float(), 'xs':xs0, 'ys':ys0, 'centers':centers}
  if 'reg' in output:
    reg = output['reg']
    reg = transpose_and_gather_feat(reg, indices)
    reg = reg.view(batch, K, 2)
    xs = xs0.view(batch, K, 1) + reg[:,:,0:1]
    ys = ys0.view(batch, K, 1) + reg[:,:,1:2]

  if 'wh' in output:
    wh = output['wh']
    wh = transpose_and_gather_feat(wh, indices)
    wh = wh.view(batch, K, 2)
    wh[wh<0] = 0
    bboxes = torch.cat([xs - wh[:, :, 0:1]/2,
                        ys - wh[:, :, 1:2]/2,
                        xs + wh[:, :, 0:1]/2,
                        ys + wh[:, :, 1:2]/2], dim=2) #[batch, K, 4]
    ret['bboxes'] = bboxes

  regression_heads = ['dep', 'rot', 'dim', 'amodel_offset', 'dep_uncertainty',
                      'latd']
  for head in regression_heads:
    if head in output:
      ret[head] = transpose_and_gather_feat(output[head], indices).view(batch, K, -1)

  return ret
