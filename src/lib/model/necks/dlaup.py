import torch
import torch.nn as nn
import numpy as np
import math

BN_MOMENTUM = 0.1

def init_weights(up):
  #[out_channel, in_channel, k, k]
  #weights like normal distribution in each kernel
  w = up.weight.data
  f = math.ceil(w.size(2) / 2)
  c = (2*f - 1 - f%2) / (2. * f)
  for i in range(w.size(2)):
    for j in range(w.size(3)):
      w[0,0,i,j] = (1-math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
  for c in range(1, w.size(0)):
      w[c,0,:,:] = w[0,0,:,:]


class ConvBNReLU(nn.Module):
  def __init__(self, in_plane, out_plane):
    super().__init__()
    self.conv = nn.Conv2d(in_plane, out_plane, kernel_size=1, 
                          stride=1, bias=False)
    self.bn = nn.BatchNorm2d(out_plane, momentum=BN_MOMENTUM)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x



class IDAUp(nn.Module):
  def __init__(self, out_channel, channel_list, scale_list,
               node_type, upsample_node):
    super().__init__()
    #aggregate channels_list[0:end]
    for i in range(1, len(channel_list)):
      c = channel_list[i]
      f = int(scale_list[i])

      conv_prev = node_type(c, out_channel)
      conv_post = node_type(out_channel, out_channel)

      #transpose conv with kernel_size=scale_list[i]*2, stride=scale_list[i]
      #dimension w/h decrease to half
      if upsample_node != 'ConvTranspose':
        raise ValueError('Unsupported upsample_node!!!')

        #Hout​=(Hin​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        #Hout=(Hin-1)*f - 2*f//2 + (f*2-1) + 1
        #比如Hin=16, 则Hout=32
      up = nn.ConvTranspose2d(out_channel, out_channel, f*2, stride=f, 
                                padding=f//2, output_padding=0,
                                groups=out_channel, bias=False)
      init_weights(up)

      setattr(self, 'conv_prev_'+str(i), conv_prev)
      setattr(self, 'up_'+str(i), up)
      setattr(self, 'conv_post_'+str(i), conv_post)

  def forward(self, layers, start_level, end_level):
      for i in range(start_level + 1, end_level):
        upsample = getattr(self, 'up_'+str(i-start_level))
        conv_prev = getattr(self, 'conv_prev_'+str(i-start_level))
        conv_post = getattr(self, 'conv_post_'+str(i-start_level))
        x = layers[i]
        x = conv_prev(x) 
        x = upsample(x)  
        layers[i] = x
        layers[i] = conv_post(layers[i]+layers[i-1])



class DLAUp(nn.Module):
  def __init__(self, first_level, channels, scales, node_type,
               upsample_node):
    super().__init__()
    self.first_level = first_level
    #self.channels = channels
    channels = list(channels)
    in_channels = channels
    scales = np.array(scales, dtype=int)

    for i in range(len(channels)-1):
      j = -i-2  #[-2,-3,-4]
      setattr(self, 'ida_{}'.format(i), 
              IDAUp(channels[j], in_channels[j:],
                    scales[j:] // scales[j],
                    node_type=node_type,
                    upsample_node=upsample_node))
      scales[j + 1:] = scales[j]
      in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

  def forward(self, layers):
    out = [layers[-1]]
    for i in range(len(layers)-self.first_level-1):
      ida = getattr(self, 'ida_{}'.format(i))
      ida(layers, len(layers)-i-self.first_level, len(layers))
      out.insert(0, layers[-1])
    return out
      


class DLASeg(nn.Module):
  """
  x = [[16, 32, 256, 256],[16, 16, 256, 256],[16, 24, 128, 128],
       [16, 32, 64, 64],[16, 96, 32, 32],[16, 320, 16, 16]]

  0. x[-1]直接压入结果
  1. 将x[5]=[16,320,16,16]先卷积为[16,96,16,16]，再反卷积为[16,96,32,32], 
     再和x[4]按位叠加, 再做一次卷积为[16,96,32,32]， 赋值给x[5]
     相当于将x[4]和x[5]信息融合
     [[16, 32, 256, 256],[16, 16, 256, 256],[16, 24, 128, 128],
      [16, 32, 64, 64],[16, 96, 32, 32],[16, 96, 32, 32]]
  2. 再将x[4]=[16,96,32,32]先卷积为[16,32,32,32]，再反卷积为[16,32,64,64]，
     再和x[3]按位叠加后做一次卷积为[[16, 32, 64, 64], 赋值给x[4]
     同理再将x[5]做一次，赋值给x[5]
     相当于x[4]和x[3]融合，再x[5]和x[4]融合
     [[16, 32, 256, 256],[16, 16, 256, 256],[16, 24, 128, 128],
      [16, 32, 64, 64],[16, 32, 64, 64],[16, 32, 64, 64]]
  3. 同理再来一次x[3]...
     [[16, 32, 256, 256],[16, 16, 256, 256],[16, 24, 128, 128],
      [16, 24, 128, 128],[16, 24, 128, 128],[16, 24, 128, 128]]
  4. 最后，x[2:5]做一次
     即现将x[3]和x[2]融合，再x[4]和x[3]融合,x[5]和x[4]融合
  5. 最后输出x[-1]=[16, 24, 128, 128]
  """
  def __init__(self, channels=None, opt=None):
    super().__init__()
    self.opt = opt
    self.channels = channels  #backbone outputs list
    self.node_type = ConvBNReLU
    self.upsample_node = opt.upsample_node if opt is not None else ''

    down_ratio = 4
    #channels[2:5] layers aggregation
    self.first_level = 2
    self.last_level = 5
    scales = [2**i for i in range(len(channels[self.first_level:]))] #[1,2,4,8]
    
    self.out_channel = channels[self.first_level]   #layer with biggest channels
    self.out_stride = [4,8,16,32]

    self.dla_up = DLAUp(self.first_level, channels[self.first_level:],
                        scales,node_type=self.node_type,
                        upsample_node=self.upsample_node)
    
    scales = [2**i for i in range(len(channels[self.first_level:self.last_level]))] #[1,2,4,8]
    self.ida_up = IDAUp(self.out_channel, channels[self.first_level:self.last_level],
                        scales, node_type=self.node_type,
                        upsample_node=self.upsample_node)

    self.out_channel = channels[self.first_level] #24

  def forward(self, x):
    y = []
    x = self.dla_up(x)
    for i in range(self.last_level - self.first_level):
        y.append(x[i].clone())
    self.ida_up(y, 0, len(y))
    return y[-1]



if __name__ == '__main__':
  x1 = torch.FloatTensor(np.random.rand(16, 32, 256, 256))
  x2 = torch.FloatTensor(np.random.rand(16, 16, 256, 256))
  x3 = torch.FloatTensor(np.random.rand(16, 24, 128, 128))
  x4 = torch.FloatTensor(np.random.rand(16, 32, 64, 64))
  x5 = torch.FloatTensor(np.random.rand(16, 96, 32, 32))
  x6 = torch.FloatTensor(np.random.rand(16, 320, 16, 16))
  x = [x1,x2,x3,x4,x5,x6]
  
  import os
  import sys
  currentdir = os.path.dirname(os.path.realpath(__file__))
  parentdir = os.path.dirname(currentdir)
  parentdir = os.path.dirname(parentdir)
  parentdir = os.path.dirname(parentdir)
  sys.path.append(parentdir)
  from opt import Opts
  
  opt = Opts()
  channels = [32, 16, 24, 32, 96, 320]
  first=2
  last=5
  scale_list=[1,2,4,8]
  dla = DLASeg(channels=channels, opt=opt)
  print(dla)
  y = dla(x)
  print(y.shape)
