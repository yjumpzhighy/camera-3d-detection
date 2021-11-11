import os
import random
import sys
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from tensorflow.python.client import device_lib

from opt import Opts
#sys.path.append("..")

from lib.dataset.dataset_factory import get_dataset
from lib.model.model import create_model, save_model
from lib.trainer import Trainer

def get_optimizer(opt, model):
  if opt.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  return optimizer

def main(opts):
  Dataset = get_dataset(opts.dataset)
  opts = Opts().update_with_dataset(opts, Dataset)
  opts.device = torch.device('cuda' if len(opts.gpus)>0 else 'cpu')

  #create model
  model = create_model(opts.arch, opts.heads, opts.head_conv, opts)

  #batch = {}
  #x = np.random.rand(16,3,512,512)
  #x = torch.FloatTensor(x)
  #batch['image'] = x
  #y = model(batch)
  #for key in y:
  #  print(key, y[key].shape)
  
  optimizer = get_optimizer(opts, model)
  lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                  milestones=opts.lr_step, gamma=0.1)
  
  start_epoch = 0
  #if load_model_path != '':
  #  model, optimizer, lr_schedule, start_epoch = load_model(..)

  dataset = Dataset(opts, opts.split)
  # train_loader = MultiEpochsDataLoader()
  dataloader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=True,
      num_workers=opts.num_workers, pin_memory=True, drop_last=True, collate_fn=None)
  
  trainer = Trainer(opts, model, optimizer)
  trainer.set_device(opts.gpus, opts.device)
  for epoch in range(start_epoch+1, opts.epochs+1):
    #run one epoch data 
    log_dict = trainer.train(epoch, dataloader)
    if epoch in opts.save_point:
      save_model(opts, f'model_{epoch}.pth', epoch, model, optimizer, lr_schedule)



if __name__ == '__main__':
  CUDA_VISIBLE_DEVICES = '0'
  #local_devices = device_lib.list_local_devices()
  #local_devices = [x.name for x in local_devices if x.device_type=='GPU']
  
  opts = Opts()
  os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_list   
  torch.manual_seed(opts.seed)
  main(opts)
