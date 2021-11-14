
import numpy as np

def alpha2roty(alpha, x, cx, fx):
  """
  alpha: observation angle of object, ranging from [-pi, pi]
  x: object center x to the camera center, (x-W/2), in pixels
  """
  #correspond to alpha/roty relations in training data
  roty = alpha + np.arctan2(x-cx, fx)  
  if roty > np.pi:
    roty -= 2*np.pi  #[-pi, pi]
  if roty < -np.pi:
    roty += 2*np.pi
  return roty


def unproject_2d_to_3d(pt_2d, dep, calib, distort=None):
  #image coordinate to camera coordinate
  #note, not for fisheye
  """
  pt_2d:2
  dep:1
  calib:3*4
  return:3
  """
  fx = calib[0][0]
  fy = calib[1][1]
  cx = calib[0][2]
  cy = calib[1][2]

  if distort is not None:
    raise ValueError('ignore distort for now please.')
  else:
    x = (pt_2d[0]-cx) / fx
    y = (pt_2d[1]-cy) / fy
    x = x * dep
    y = y * dep
    pt_3d = np.array([x,y,dep], dtype=np.float32).reshape(3)
  return pt_3d


def ddd2locrot(center, alpha, dim, depth, calib, amodel_center):

  locations = unproject_2d_to_3d(amodel_center,depth,calib)
  locations[1] += dim[0]/2  #correspond to training data
  roty = alpha2roty(alpha, center[0], calib[0,2], calib[0,0])
  return locations, roty
