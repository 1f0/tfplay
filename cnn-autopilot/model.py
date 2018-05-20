import tensorflow as tf

def weight_var(shape):
	init = tf.truncated_normal(shape, stddev=.1)
	return tf.Variable(init)

def bias_var(shape):
	init = tf.constant(0.1, shape=shape)
	return tf.Variable(init)

def wb(shape):
	return weight_var(shape), bias_var(shape[-1:])

def conv2d(x, w, stride):
	strides = [1, stride, stride, 1]
	return tf.nn.conv2d(x,w,strides=strides,padding='VALID')

def conv_layer(x, shape, stride):
	w_conv, b_conv = wb(shape)
	return tf.nn.relu(conv2d(x, w_conv, stride)+b_conv)

def fcl(x, shape, keep_prob):
	w, b = wb(shape)
	fc = tf.nn.relu(tf.matmul(x, w)+b)
	return tf.nn.dropout(fc, keep_prob)

x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

def my_model():
	# five layer cnn
	c1 = conv_layer(x, [5,5,3,24],2)
	c2 = conv_layer(c1, [5,5,24,36],2)
	c3 = conv_layer(c2, [5,5,36,48],2)
	c4 = conv_layer(c3, [3,3,48,64],1)
	c5 = conv_layer(c4, [3,3,64,64],1)
	
	flat = tf.reshape(c5, [-1,1152])	

	# FCL
	f1 = fcl(flat, [1152, 1164], keep_prob)
	f2 = fcl(f1, [1164, 100], keep_prob)
	f3 = fcl(f2, [100, 50], keep_prob)
	f4 = fcl(f3, [50, 10], keep_prob)
	
	# output
	w, b = wb([10, 1])
	return tf.multiply(tf.atan(tf.matmul(f4,w) + b), 2)

y_ = my_model()
