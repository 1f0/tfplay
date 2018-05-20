from crnn import model, ctc_model
from utils import alphabet, input_shape


def predict(im):
  X = np.array([im.reshape((height, width, 1))])
  y_pred = model.predict(X)
  y_pred = y_pred[:, 2:, :]
  code = pred.argmax(axis=2)[0]
  print([alphabet[c] for c in code])
