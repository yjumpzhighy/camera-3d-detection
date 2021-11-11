import torch
import torch.nn as nn
import os

from .generic_net import GenericNet


network_factory = {'res': None, 
                   'generic':GenericNet
                  }

def create_model(arch, head, head_conv, opt=None):
  num_layers = int(arch[arch.find('_')+1:] if '_' in arch else 0)  #res_101
  arch = arch[:arch.find('_')] if '_' in arch else arch
  model_class = network_factory[arch]
  model = model_class(num_layers, head, head_conv, opt)
  return model
 
def save_model(opt, model_name, epoch, model, optimizer, lr_schedule):
  path = os.path.join(opt.save_dir, model_name)
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch':epoch, 'state_dict':state_dict}
  if optimizer is not None:
    data['optimizer'] = optimizer.state_dict()
  if lr_schedule is not None:
    data['lr_schedule'] = lr_schedule.state_dict()
  torch.save(data, path)
  
