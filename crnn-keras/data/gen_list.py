import os
import sys
from random import shuffle

sys.path.append("..")
from config import n_train, n_val, n_test

folder = sys.argv[1]
all = os.listdir(folder)
shuffle(all)

tmp = n_train + n_val
train = all[:n_train]
val = all[n_train:tmp]
test = all[tmp:tmp + n_test]


def write_file(name, arr):
  with open(name, 'w') as f:
    for s in arr:
      label = s.split('_')[0]
      line = '%s/%s %s\n' % (folder, s, label)
      f.write(line)

write_file('train.txt', train)
write_file('test.txt', test)
write_file('valid.txt', val)
