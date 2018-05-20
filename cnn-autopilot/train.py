import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2

# my load data & model
import load
import model

sess = tf.InteractiveSession()
train_vars = tf.trainable_variables()

regular_term = tf.add_n([tf.nn.l2_loss(v) for v in train_vars])*0.001
predict_loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))
loss = predict_loss + regular_term

train_step = tf.train.AdamOptimizer(2e-3).minimize(loss)
sess.run(tf.initialize_all_variables())

tf.summary.scalar("loss", loss)
merged_sum = tf.summary.merge_all()
saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)

train_logs = './train_logs'
test_logs = './test_logs'
train_summary_writer = tf.summary.FileWriter(train_logs, graph=tf.get_default_graph())
test_summary_writer = tf.summary.FileWriter(test_logs)

epochs = 20 
batch_size = 100

for epoch in range(epochs):
	for i in range(int(load.num_images/batch_size)):
		x,y = load.LoadTrainBatch(batch_size)
		train_step.run(feed_dict={model.x: x, model.y: y, model.keep_prob:0.8})
		step_num = epoch * load.num_images/batch_size + i
		summary = merged_sum.eval(feed_dict={model.x: x, model.y: y, model.keep_prob:1.0})
		train_summary_writer.add_summary(summary, step_num)

		if i%10 == 0:
			x, y = load.LoadValBatch(batch_size)
			loss_val = loss.eval(feed_dict={model.x: x, model.y: y, model.keep_prob:1.0})
			print("Epoch: %d, Step: %d, Loss %g" %(epoch, step_num, loss_val))		
			summary = merged_sum.eval(feed_dict={model.x: x, model.y: y, model.keep_prob:1.0})
			test_summary_writer.add_summary(summary, step_num)

	save_dir = './save'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	checkpoint = os.path.join(save_dir, "model.ckpt")
	name = saver.save(sess, checkpoint)
	print("Model saved in file: %s" % name)

print("Run 'tensorboard --logdir=./logs'. Then open http://0.0.0.0:6006/")
