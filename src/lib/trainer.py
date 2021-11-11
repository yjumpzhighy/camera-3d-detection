import torch
import torch.nn as nn

import time

from .model.loss import GenericLoss
from .utils.utils import AverageMeter


class ModelWithLoss(nn.Module):
  def __init__(self, model, losses, opt):
    super().__init__()
    self.model = model
    self.losses = losses
    self.opt = opt

  def forward(self, batch, iter_id):
    #Run model and get predictions
    outputs = self.model(batch)
    losses, loss_stats, losses_gn = self.losses(outputs, batch)
    return outputs, losses, loss_stats, losses_gn




class Trainer(object):
  def __init__(self, opt, model, optimizer):
    self.opt = opt
    self.model = model
    self.optimizer = optimizer
    self.loss_states, self.losses = self._get_losses()
    self.model_with_loss = ModelWithLoss(model, self.losses, opt)
    print('==>Trainer set up, ready to train...')

  def _get_losses(self):
    loss_order = {'hm', 'wh', 'reg', 'dep', 'dim', 'rot', 'latd', \
                  'amodel_offset', 'depth_uncertain'}
    loss_states = ['tot'] + [k for k in loss_order if k in self.opt.heads]
    loss = GenericLoss(self.opt)
    return loss_states, loss

  def set_device(self, gpus, device):
    #copy model
    #if (len(gpus)>1):
    #  self.model_with_loss=DataParallel(self.model_with_loss, device_ids=gpus, \
    #                         chunk_sizes=chunk_sizes).to(device)
    #else:
    #  self.model_with_loss = self.model_with_loss.to(device)
    self.model_with_loss.to(device)
      
    #copy optimizer
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def train(self, epoch, data_loader):
    return self._run_epoch('train', epoch, data_loader)

  def _run_epoch(self, split, epoch, data_loader):
      if split == 'train':
        self.model_with_loss.train()  #set model train mode,effect dropout, BN, etc
      else:
        self.model_with_loss.eval()  #set model eval mode,effect dropout, BN, etc
      torch.cuda.empty_cache()

      results = {}
      data_load_time = AverageMeter()
      data_move_time = AverageMeter()
      batch_time = AverageMeter()
      fw_time = AverageMeter()
      bw_time = AverageMeter()
      avg_loss_stats = {l: AverageMeter() for l in self.loss_states \
                        if l=='tot' or self.opt.weights_dict[l]>0}

      end_time = time.time()
      #start process
      self.optimizer.zero_grad()
      for iter_id, batch in enumerate(data_loader):
        print('iter id:', iter_id)
        data_load_time.update(time.time() - end_time)
        data_move_end = time.time()
        for k in batch:
          if type(batch[k])==dict:
            batch[k] = {key: val.to(device=self.opt.device, non_blocking=True) \
                        for key, val in batch[k].items()}
          elif type(batch[k])==list:
            for j in range(len(batch[k])):
              if type(batch[k][j])==dict:
                batch[k][j] = {key: val.to(device=self.opt.device, non_blocking=True) \
                               for key, val in batch[k][j].items()}
          else:
            batch[k] = batch[k].to(device=self.opt.device, non_blocking=True)
        data_move_time.update(time.time() - data_move_end)

        fw_end = time.time()
        #run model and loss
        output, losses, loss_stats, losses_gn = self.model_with_loss(batch, iter_id)
        loss = losses.mean()
        print('loss:', loss)
        print('---------------------------')
        fw_time.update(time.time() - fw_end)
        bw_end = time.time()

        #back propogation
        if split=='train':
          loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

        bw_time.update(time.time() - bw_end)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        for l in avg_loss_stats:
          avg_loss_stats[l].update(loss_stats[l].mean().item(), \
          batch['image'].size(0))

      ret = {k: v.avg for k, v in avg_loss_stats.items()}
      return ret
