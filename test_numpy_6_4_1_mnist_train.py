# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py中定义的常量和前向传播的函数
import test_numpy_6_4_1_mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 1000				# 后注：一个训练batch中的训练数据个数
LEARNING_RATE_BASE = 0.8		# 后注：训练的学习率
LEARNING_RATE_DECAY = 0.99		# 后注：学习率的衰减率
REGULARIZATION_RATE = 0.0001	# 后注：描述训练模型复杂度的正则化项在损失函数中的系数，即“J(θ)+λR(w)”中的λ
TRAINING_STEPS = 30000			# 后注：训练轮数
MOVING_AVERAGE_DECAY = 0.99		# 后注：滑动平均衰减率

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "/path/to/model"
MODEL_NAME = "model.ckpt"

def train(mnist):
	#定义输入输出placeholder
	# x = tf.placeholder(
	# 	tf.float32, [None, test_numpy_5_5_mnist_inference.INPUT_NODE], name = 'x-input')

	# 后注：输入节点矩阵为[1000, 28, 28, 1]
	x = tf.placeholder(tf.float32, [
				BATCH_SIZE, 
				test_numpy_6_4_1_mnist_inference.IMAGE_SIZE,
				test_numpy_6_4_1_mnist_inference.IMAGE_SIZE,
				test_numpy_6_4_1_mnist_inference.NUM_CHANNELS], name = 'x-input')

	y_ = tf.placeholder(
		tf.float32, [None, test_numpy_6_4_1_mnist_inference.OUTPUT_NODE], name = 'y-input')

	# 后注：计算模型的正则化损失
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

	# 直接使用mnist_inference.py中定义的前向传播过程
	y = test_numpy_6_4_1_mnist_inference.inference(x, 1, regularizer)

	# 后注：注意！！global_step传入反向传播算法函数后将被自动更新，这里只给初始值即可，见P86
	global_step = tf.Variable(0, trainable = False)

	# 和5.2.1小节样例中类似地定义损失函数、学习率、滑动平均操作以及训练过程
	# 后注：定义滑动平均的类
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	# 后注：定义一个更新变量滑动平均的操作
	variables_averages_op = variable_averages.apply(
		tf.trainable_variables())
	# 后注：计算交叉熵
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits = y, labels = tf.argmax(y_, 1))
	# 后注：计算交叉熵平均值
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	# 后注：总损失
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	# 后注：学习率
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_DECAY, 
		global_step, 
		mnist.train.num_examples / BATCH_SIZE, 
		LEARNING_RATE_DECAY)
	# 后注：优化损失函数
	train_step = tf.train.GradientDescentOptimizer(learning_rate) \
						.minimize(loss, global_step = global_step)

	with tf.control_dependencies([train_step, variables_averages_op]) :
		train_op = tf.no_op(name = 'train')

	# 初始化TensorFlow持久化类
	saver = tf.train.Saver()

	with tf.Session() as sess: 
		tf.initialize_all_variables().run()

		# 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成。
		for i in range(TRAINING_STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)

			reshaped_xs = np.reshape(xs, (BATCH_SIZE, 
											test_numpy_6_4_1_mnist_inference.IMAGE_SIZE, 
											test_numpy_6_4_1_mnist_inference.IMAGE_SIZE,
											test_numpy_6_4_1_mnist_inference.NUM_CHANNELS))

			_, loss_value, step = sess.run([train_op, loss, global_step], 
											feed_dict = {x: reshaped_xs, y_ : ys})
			# 每1000轮保存一次模型
			if i % 10 == 0:
				# 输出当前模型的情况。这里只输出了模型在当前训练batch上的损失函数大小。通过损失
				# 函数的大小可以大概了解训练的情况。在验证数据集上的正确率信息会有一个单独的程序
				# 来生成。
				print("After %d training step(s), loss on training "
						"batch is %g." % (step, loss_value))

				# 保存当前的模型。注意这里给出了global_step参数，这样可以让每个被保存模型的文件名
				# 末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮之后得到的模型
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), 
							global_step = global_step)

def main(argv = None):
	# 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
	mnist = input_data.read_data_sets("/tmp/data", one_hot = True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()