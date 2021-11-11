import torch


class __RepeatSampler(object):
  def __init__(self, sampler):
    self.sampler = sampler
  def __iter__(self):
    while True:
      yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    setattr(self, 'batch_sampler', __RepeatSampler(self.batch_sampler)
