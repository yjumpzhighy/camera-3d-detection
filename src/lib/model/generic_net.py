import torch
import torch.nn as nn

from .arch_factory import get_backbone, get_neck

def fill_fc_weights(layers):
  for m in layers.modules():
    if isinstance(m, nn.Conv2d):
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)


class GenericNet(nn.Module):
  def __init__(self, num_layers, heads, head_convs, opt=None, num_stacks=1):
    super(GenericNet, self).__init__()
    print('==> Use generic model with backbone {} and neck {}'.format(
          opt.backbone, opt.neck))
    head_kernel = opt.head_kernel
    self.opt = opt
    
    backbone = get_backbone(opt.backbone)
    self.backbone = backbone(opt=opt)
    neck = get_neck(opt.neck)
    self.neck = neck(channels=self.backbone.channels, opt=self.opt)
    self.out_channel = self.neck.out_channel
    self.heads = heads
    self.head_convs = head_convs

    for head in self.heads:
      classes = self.heads[head]  #output number of head
      head_conv = self.head_convs[head]  #input channels num from neck

      conv1 = nn.Conv2d(self.out_channel, head_conv[0], kernel_size=head_kernel,
                        padding=head_kernel//2, bias=True)
      conv2 = nn.Conv2d(head_conv[0], classes, kernel_size=1, stride=1,
                        padding=0, bias=True)
      fc = nn.Sequential(conv1, nn.ReLU(inplace=True), conv2)

      fill_fc_weights(fc)
      setattr(self, head, fc)

  def forward(self, x):
    img = x['image']
    x = self.backbone(img)
    x = self.neck(x)
    out={}
    for head in self.heads:
      fc = getattr(self, head)
      out[head] = fc(x)
    return out
  
