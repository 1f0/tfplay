import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow
import sys

if sys.argv is None or len(sys.argv) < 3:
	print('usage: visual.py input_image output_image')
	sys.exit()

im_size = [66, 200]
im = imresize(imread(sys.argv[1])[-150:], im_size)/255.0
data = [im]
nodes = []

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('save/model.ckpt.meta')
	saver.restore(sess, 'save/model.ckpt')
	graph = tf.get_default_graph()

	# forward
	x = graph.get_tensor_by_name('Placeholder:0')
	c = graph.get_tensor_by_name('Relu:0')
	nodes.append(sess.run(c, feed_dict={x:data}))

	for i in range(1, 5):
		c = graph.get_tensor_by_name('Relu_%d:0' % i)
		nodes.append(sess.run(c, feed_dict={x:data}))

# backward
for i in range(1, 6):
	avg = np.mean(nodes[5-i], axis=3)[0]
	if i == 1:
		tmp = avg
	else:
		tmp = np.multiply(avg, tmp)
	if i != 5:
		tmp = imresize(tmp, nodes[5-(i+1)].shape[1:3])

tmp = imresize(tmp, im.shape)
min = np.min(tmp)
max = np.max(tmp)
tmp = (tmp-min)/(max-min)

im[:,:,1] = tmp + im[:,:,1]
imsave(sys.argv[2], im)
