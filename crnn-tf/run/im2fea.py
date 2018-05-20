from __future__ import print_function
import os
import os.path as ops
import argparse
import numpy as np
import cv2

from run.feaio import FeatureIO


def init_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--text', type=str,
                      help='images and labels list file')
  parser.add_argument('--save_dir', type=str, help='saved filename')

  return parser.parse_args()


def write_features(anno, save_dir):
  if not ops.exists(save_dir):
    os.makedirs(save_dir)

  print('Opening anno.txt ...')
  with open(anno, 'r') as anno_file:
    info = np.array([tmp.strip().split() for tmp in anno_file.readlines()])
    images = np.array([cv2.resize(cv2.imread(ops.join(ops.dirname(anno), tmp), cv2.IMREAD_GRAYSCALE),(100,32))
                       for tmp in info[:, 0]])
    labels = np.array([tmp for tmp in info[:, 1]])
    imagenames = np.array([ops.basename(tmp) for tmp in info[:, 0]])

  print('Open anno.txt finished')

  def shuffle_images_labels(images, labels, imagenames):
    images = np.array(images)
    labels = np.array(labels)

    assert images.shape[0] == labels.shape[0]

    random_index = np.random.permutation(images.shape[0])
    shuffled_images = images[random_index]
    shuffled_labels = labels[random_index]
    shuffled_imagenames = imagenames[random_index]

    return shuffled_images, shuffled_labels, shuffled_imagenames

  images, labels, imagenames = shuffle_images_labels(
      images, labels, imagenames)

  def preprocess_imgs(imgs):
    imgs = [np.repeat(tmp[..., None], 3, axis=2) for tmp in imgs]
    imgs = [bytes(list(np.reshape(tmp, [100 * 32 * 3])))
            for tmp in imgs]
    return imgs

  images = preprocess_imgs(images)

  writer = FeatureIO().writer
  anno_name = ops.splitext(ops.basename(anno))[0]
  tfrecord_path = ops.join(save_dir, anno_name + '.tfrecords')
  writer.write_features(tfrecords_path=tfrecord_path, labels=labels, images=images,
                        imagenames=imagenames)

if __name__ == '__main__':
  args = init_args()
  if not ops.exists(args.text):
    raise ValueError('Text doesn\'t exist')
  write_features(anno=args.text, save_dir=args.save_dir)
