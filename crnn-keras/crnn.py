from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras.layers import Reshape, LSTM, Bidirectional, Dense, GRU
from keras.layers import Input, Lambda, ZeroPadding2D, Permute, TimeDistributed
from keras import backend as K

from utils import alphabet, input_shape
n_class = len(alphabet)


def convRelu(i, bn=False):
  nm = [64, 128, 256, 256, 512, 512, 512]  # filter
  ks = [3, 3, 3, 3, 3, 3, 2]  # kernel size
  pn = [1, 1, 1, 1, 1, 1, 0]  # padding number
  ps = ['same' if x == 1 else 'valid' for x in pn]

  if i == 0:
    return Conv2D(nm[i], ks[i],
                  padding=ps[i],
                  activation='relu',
                  kernel_initializer='he_normal',
                  input_shape=input_shape
                  )

  else:
    return Conv2D(nm[i], ks[i],
                  padding=ps[i],
                  activation='relu',
                  kernel_initializer='he_normal'
                  )

model = Sequential()
# cnn
model.add(convRelu(0))
model.add(MaxPooling2D(pool_size=2))  # strides=pool, 16x50x64
model.add(convRelu(1))
model.add(MaxPooling2D(pool_size=2))  # 8x25x128
model.add(convRelu(2))
model.add(convRelu(3))
model.add(MaxPooling2D(pool_size=2, strides=(2, 1), padding='same'))  # 4x25x256
model.add(convRelu(4))
model.add(BatchNormalization(axis=1))
model.add(convRelu(5))
model.add(BatchNormalization(axis=1))
model.add(ZeroPadding2D(padding=(0, 1)))  # 4x27x512
model.add(MaxPooling2D(pool_size=2, strides=(
    2, 1), padding='valid'))  # 2x26x512
model.add(convRelu(6))  # 1x25x512
model.add(Permute((2, 1, 3)))  # 25x1x512, in case height != 32
model.add(TimeDistributed(Flatten()))  # 25x512

# map to sequence, bi-lstm
model.add(Bidirectional(LSTM(256, dropout=0.5, return_sequences=True)))
model.add(Bidirectional(LSTM(256, dropout=0.5, return_sequences=True)))
model.add(Dense(n_class, kernel_initializer='he_normal',
                activation='softmax'))  # 25x64

model.summary()
import sys
sys.exit()

def ctc_func(args):
  y_pred, labels, input_length, label_length = args
  # the 2 is critical here since the first couple outputs of the RNN
  # tend to be garbage:
  y_pred = y_pred[:, 2:, :]
  return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

input_data = Input(name='the_input', shape=input_shape, dtype='float32')
y_pred = model(input_data)

labels = Input(name='the_labels', shape=[None, ], dtype='int32')
input_length = Input(name='input_length', shape=[1], dtype='int32')
label_length = Input(name='label_length', shape=[1], dtype='int32')

ctc_loss = Lambda(ctc_func, name='ctc', output_shape=(1,))(
    [y_pred, labels, input_length, label_length])
ctc_model = Model(
    inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)
ctc_model.summary()
ctc_model.compile(
    loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
