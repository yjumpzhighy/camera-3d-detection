import cv2
import numpy as np

def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
  """
  get the affine from input [W,H] to output [W',H'] with W'=W/4, H'=H/4
  center = [W/2, H/2]
  scale = max(H, W)
  output_size = [W', H']
  """

  if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
    scale = np.array([scale, scale], dtype=np.float32)

  scale_tmp = scale
  src_w = scale_tmp[0]
  dst_w = output_size[0]
  dst_h = output_size[1]

  rot_rad = np.pi * rot / 180
  src_dir = get_dir([0, src_w * -0.5], rot_rad)
  dst_dir = get_dir([0, dst_w * -0.5], rot_rad)

  src = np.zeros((3,2), dtype = np.float32)
  dst = np.zeros((3,2), dtype = np.float32)
  src[0,:] = center
  dst[0,:] = [dst_w / 2, dst_h / 2]
  src[1,:] = center + src_dir
  dst[1,:] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
  src[2:, :] = get_3rd_point(src[0, :], src[1, :])
  dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

  if inv:
    trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
  else:
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
  return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]



def transform_preds_with_trans(coords, trans):
  """
  coords = [k,2]
  """
  target_coords = np.ones((coords.shape[0],3), np.float32)
  target_coords[:, :2] = coords
  target_coords = np.dot(trans, target_coords.transpose()).transpose()
  return target_coords[:, :2]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
  #rotate src_point by rot_rad with z-axis, right hand.
  sn, cs = np.sin(rot_rad), np.cos(rot_rad)
  result = [0, 0]
  result[0] = src_point[0] * cs - src_point[1] * sn
  result[1] = src_point[0] * sn + src_point[1] * cs
  return result

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size
  a1 = 1
  b1 = height + width
  c1 = width * height * (1-min_overlap) / (1+min_overlap)
  sq1 = np.sqrt(b1**2 - 4*a1*c1)
  r1 = (b1+sq1)/2

  a2 = 4
  b2 = 2 * (height + width)
  c2 = width * height * (1-min_overlap)
  sq2 = np.sqrt(b2**2 - 4*a2*c2)
  r2 = (b2+sq2)/2

  a3 = 4 * min_overlap
  b3 = -2 * min_overlap * (height + width)
  c3 = width * height * (min_overlap-1)
  sq3 = np.sqrt(b3**2 - 4*a3*c3)
  r3 = (b3+sq3)/2

  return min(r1,r2,r3)

def gaussian2D(shape, sigma=1):
  """
  return (shape[0], shape[1]) gaussian kernel
  """
  m,n = [(ss-1.)/2. for ss in shape]
  #similar with np.arange(-m,m+1), but format is 2d matrix
  y,x = np.ogrid[-m:m+1, -n:n+1]
  h = np.exp(-(x*x+y*y)/(2*sigma*sigma)) #[shape[0], shape[1]]
  h[h<np.finfo(h.dtype).eps * h.max()] = 0
  return h



def draw_umich_gaussian(heatmap, center, radius, k=1):
  """
  heatmap: gt heatmap based on specific category [H', W']
  center: bbox center
  radius: bbox gt gaussian radius
  """
  diameter = 2*radius+1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter/6)
  x,y = int(center[0]), int(center[1])
  h,w = heatmap.shape[0:2] #equals to net output size
  left,right = min(x, radius), min(w-x,radius+1)
  top,bottom = min(y, radius), min(h-y,radius+1)  #boundary constraint for next area select

  masked_heatmap = heatmap[y-top:y+bottom, x-left:x+right]
  masked_gaussian = gaussian[radius-top:radius+bottom, radius-left:radius+right]

  #overlayer gaussian kernel distribution onto heatmap
  if min(masked_gaussian.shape)>0 and min(masked_heatmap.shape)>0:
    np.maximum(masked_heatmap, masked_gaussian*k, out=masked_heatmap)

  return heatmap
