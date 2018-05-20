from string import ascii_letters, digits, ascii_lowercase
# alphabet = ascii_letters + digits + ' '
alphabet = ascii_lowercase + digits + ' '
r = range(len(alphabet))
dictionary = dict(zip(alphabet, r))
height = 32
width = 100
input_shape = (height, width, 1)

import numpy as np
import cv2


def to_fix_height(im, w=None):
  im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
  if w is None:
    scale = im.shape[0] * 1.0 / height
    w = int(im.shape[1] / scale)
  im = cv2.resize(im, (w, height))
  return im


def load_images(path):
  im = cv2.imread(path)
  im = to_fix_height(im, width)
  im = im.astype(np.float32)
  im = im.reshape(input_shape)
  im -= 128.0
  im /= 128.0
  return im
