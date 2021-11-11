import torch
import torch.nn as nn
import os
import numpy as np

def make_divisible(channels, divisor):
  """
  Ensures that all layers have a channels number that is divisible by divisor:8
  """
  min_value = divisor
  new_channels = max(min_value, int(channels+divisor/2) // divisor * divisor)
  if new_channels < 0.9 * channels:
    # Make sure that round down does not go down by more than 10%.
    new_channels += divisor
  return new_channels


class ConvBNReLU(nn.Module):
  def __init__(self, in_plane, out_plane, kernel_size=3, stride=1, groups=1):
    super(ConvBNReLU, self).__init__()
    padding = (kernel_size-1)//2
    self.conv = nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding,
                          groups=groups, bias=False)
    self.bn = nn.BatchNorm2d(out_plane)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x 


class InvertedResidual(nn.Module):
  # increase channels first, then decrease 
  def __init__(self, in_plane, out_plane, stride, expand_ratio):
    super().__init__()
    self.inplane = in_plane
    self.outplane = out_plane
    self.stride = stride
    self.expand_ratio = expand_ratio
    hidden_layer_chan = int(round(in_plane*expand_ratio))

    layers = []
    if expand_ratio != 1:
      #step1, point-wise conv to expand channels 
      layers.append(ConvBNReLU(in_plane, hidden_layer_chan, kernel_size=1))
    #step2, depth-wise conv 
    layers.append(ConvBNReLU(hidden_layer_chan, hidden_layer_chan, 
                             stride=self.stride, groups=hidden_layer_chan))
    #step3, point-wise conv to shrink channels
    layers.append(nn.Conv2d(hidden_layer_chan, out_plane, 1, 1, 0, bias=False))
    layers.append(nn.BatchNorm2d(out_plane))
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    out = self.layers(x)
    if (self.stride==1 and self.inplane==self.outplane):
      #must be same dimension to add identity
      out += x
    return out



class MobileNetV2(nn.Module):
  def __init__(self, opt=None, width_mult=1.0, round_nearest=8):
    """
    Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
    """
    super(MobileNetV2, self).__init__()
    self.opt = opt
    block = InvertedResidual
    input_channel = 32
    #input_channel = make_divisible(input_channel * width_mult, round_nearest)
    last_channel = 1280

    inverted_res_settings = [
      #expand ratio, output channels, repeat, stride
      [1,16,1,1],
      [6,24,2,2],
      [6,32,3,2],
      [6,64,4,2],
      [6,96,3,1],
      [6,160,3,2],
      [6,320,1,1],
    ]

    if opt is not None and opt.use_coord_input:
      input_dim = 5
    else:
      input_dim = 3
    
    self.block_cnt = 0
    setattr(self, 'features_{}'.format(self.block_cnt), 
            ConvBNReLU(input_dim, input_channel, stride=2))

    #self.features = [ConvBNReLU(input_dim, input_channel, stride=2)]

    all_channels = [input_channel]
    self.key_blocks = [True]  #record which block output will be stream out
    #build inverted residual blocks
    for t, c, n, s in inverted_res_settings:
      output_channel = c
      for i in range(n):
        stride = s if i==0 else 1
        #self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
        self.block_cnt += 1
        setattr(self, 'features_{}'.format(self.block_cnt), 
                block(input_channel, output_channel, stride, expand_ratio=t))
        input_channel = output_channel
        all_channels.append(output_channel)
        if stride > 1:
          self.key_blocks.append(True)
        else:
          self.key_blocks.append(False)

    for i in range(len(self.key_blocks)-1):
      if self.key_blocks[i+1]:
        self.key_blocks[i] = True
        self.key_blocks[i+1] = False

    self.key_blocks[-1] = True
    
    self.channels = []
    for i in range(len(self.key_blocks)):
      if self.key_blocks[i]:
        #[32, 24, 32, 64, 160, 320]
        self.channels.append(all_channels[i])

    print('MobileNetV2 channels:', self.channels)
    

  def forward(self, inputs):
    # x = self.features[0](inputs)
    # out = [x]
    # # output output from several blocks for various level features
    # for i in range(1, len(self.features)):
    #   x = self.features[i](x)
    #   if self.key_blocks[i]:
    #     out.append(x)
    # return out

    block = getattr(self, 'features_{}'.format(0))
    x = block(inputs)
    out = [x]
    for i in range(1, self.block_cnt+1):
      block = getattr(self, 'features_{}'.format(i))
      x = block(x)
      if self.key_blocks[i]:
        out.append(x)
    return out





if __name__ == '__main__':
  t = MobileNetV2()
  print(t)
  x = np.random.rand(16,3,512,512)
  x = torch.FloatTensor(x)
  x = t(x)
  for l in x:
    print(l.shape)
