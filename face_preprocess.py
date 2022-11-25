import cv2
import numpy as np
from skimage import transform as trans

def align_face(img, line, face_template_size, image_size):
        line_vec = line.strip().split(' ')
        if len(line_vec) != 14:
            return
        if img is None:
            return
        if img.ndim == 2:
            img = to_rgb(img)
        points_vec = [float(e) for e in line_vec[4::]]
        _landmark = np.array(points_vec)
        _landmark = _landmark.reshape(2,5).T
        warped = preprocess(img, _landmark, face_template_size, image_size)

        return warped    

def preprocess(img, landmark, face_template_size, image_size):
  M = None
  if landmark is not None:
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32)
    src[:,0] *= float(face_template_size[0] / 96)
    src[:,0] += (image_size[0]-face_template_size[0])/2
    src[:,1] *= float(face_template_size[1] / 112)
    src[:,1] += (image_size[1]-face_template_size[1])/2
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

  if M is None:
    return None
  warped = cv2.warpAffine(img,M,image_size, borderMode=cv2.BORDER_REPLICATE)
    
  return warped