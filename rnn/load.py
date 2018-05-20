import sys
import numpy as np

if len(sys.argv) < 2:
  print('usage: %s loadfile' %sys.argv[0])
  sys.exit()

with open(sys.argv[1], 'r') as f:
  text = f.read().decode('utf-8')

words = set(text)
#enumerate is generator, can be used just once
word_to_id = {c: i for i, c in enumerate(words)}
id_to_word = dict(enumerate(words))

encoded = np.array([word_to_id[c] for c in text], dtype=np.int32)

