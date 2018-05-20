from load import words, word_to_id, id_to_word, encoded
import numpy  as np
import matplotlib.pyplot as plt

hist, bin_edges = np.histogram(encoded, bins=len(words))
