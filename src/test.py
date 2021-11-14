import os
import torch
import cv2

from opt import Opts
from lib.dataset.dataset_factory import get_dataset
from lib.model.model import create_model
from lib.detector import Detector

def test(opts):
  Dataset = get_dataset(opts.dataset)
  opts = Opts().update_with_dataset(opts, Dataset)
  opts.device = torch.device('cuda' if len(opts.gpus)>0 else 'cpu')
  
  opts.split = 'test'
  dataset = Dataset(opts, opts.split)
  detector = Detector(opts)

  num_iters = len(dataset) #call dataset __len__ to get the sample number
  results = {}
  for ind in range(num_iters):
    img_id = dataset.imgIds[ind]
    img_info = dataset.coco.loadImgs([img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(dataset.img_path, file_name)
    input_meta = {}  #for other inputs info, like pre_hm,pre_img
    if 'calib' in img_info:
      input_meta['calib'] = img_info['calib']
    
    #ret = detector.run(img_path, input_meta)
    #results[img_id] = ret['results']

    results = detector.run(img_path, input_meta)

    #visualize:
    show_img = cv2.imread(img_path)
    obj_bboxes = results['results']
    for i in range(len(obj_bboxes)):
      bbox = obj_bboxes[i]['bbox']
      cv2.rectangle(show_img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), \
                        (0,255,0), 2)
    cv2.imshow('image window', show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    break

if __name__ == '__main__':
  opts = Opts()
  os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_list  
  test(opts) 
