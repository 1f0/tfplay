from crnn import ctc_model
from utils import dictionary, load_images, input_shape
from config import n_train, n_val, n_test
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import CSVLogger
import numpy as np

data_dir = 'data/'
batch_size = 1  # 2
max_label_len = 16
epochs = 3  # 50
down_sample_factor = 4


def process_line(line):
  name, labels = line.split()
  path = data_dir + name
  y = [dictionary[i] for i in labels]  # 0~nclass
  return path, y


def generate_arrays(path):
  with open(path) as f:
    X = np.zeros((batch_size,) + input_shape)
    Y = np.zeros([batch_size, max_label_len])
    input_len = np.zeros([batch_size, 1])
    label_len = np.zeros([batch_size, 1])
    while True:
      cnt = 0
      for line in f:
        path, label = process_line(line)
        X[cnt, :, :, :] = load_images(path)
        Y[cnt, 0:len(label)] = label
        # TODO: fix this hard coded
        input_len[cnt] = int(input_shape[1] / down_sample_factor) - 2
        label_len[cnt] = len(label)

        cnt += 1
        if(cnt == batch_size):
          input = [X, Y, input_len, label_len]
          output = np.zeros([batch_size, 1])
          yield input, output
          X = np.zeros((batch_size,) + input_shape)
          Y = np.zeros([batch_size, max_label_len])
          input_len = np.zeros([batch_size, 1])
          label_len = np.zeros([batch_size, 1])
          cnt = 0

if __name__ == '__main__':
  ctc_model.fit_generator(
      generate_arrays(data_dir + 'train.txt'),
      steps_per_epoch=1,#int(n_train / batch_size),
      epochs=epochs,
      validation_steps=1,#int(n_val / batch_size),
      validation_data=generate_arrays(
          data_dir + 'valid.txt'),
      callbacks=[ModelCheckpoint(
          "save/model{epoch:02d}-{val_loss:.4f}.hdf5"), TensorBoard(), CSVLogger('logs/train.csv')],
  )
