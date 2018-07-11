import tensorflow as if
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784			#输入层节点数。相当于图片像素。
OUTPUT_NODE = 10			#输出层节点数。类别的数目。

LAYER1_NODE = 500			#隐藏层节点数。使用只有一个隐藏层的网络结构作为样例，这个隐藏层有500个节点。

BATCH_SIZE = 100			#一个训练batch中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降。

LEARNING_RATE_BASE = 0.8			#基础的学习率
LEARNING_RATE_DECAY = 0.99			#学习率的衰减率

REGULARIZATION_RATE = 0.0001		#描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000				#训练轮数
MOVING_AVERAGE_DECAY = 0.99			#滑动平均衰减率


# 一个辅助函数，给定神经网络的输入的所有参数，计算神经网络的前向传播结果。在这里定义了一个使用ReLU激活函数的三层全连接神经网络。
# 通过加入隐藏层实现了多层网络结构，通过ReLU激活函数实现了去线性化。在这个函数中也支持传入用于计算参数平均值的类，
# 这样方便在测试时使用滑动平均模型
def  inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
	# 当没有提供滑动平均类时，直接使用参数当前的取值。
	if avg_class == None:
		# 计算隐藏层的前向传播结果，这里使用了ReLU激活函数
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

		# 计算输入层的前向传播结果。因为在计算损失函数时会一并计算softmax函数，所以这里不需要加入激活函数。
		# 而且不加入softmax不会影响预测结果。因为预测时使用的是不同类别对应节点输出值的相对大小，
		# 有没有softmax层对最后分类结果的计算没有影响。于是在计算整个神经网络的前向传播时可以不加入最后的softmax层
		return tf.matmul(layer1, weights2) + biases2

	else:
		# 首先使用avg_class.average函数来计算得出变量的滑动平均值
		layer1 = tf.nn.relu(
			tf.matmul(input_tensor, avg_class.average(weights1)) + 
			avg_class.average(biases1))
		return tf.matmul(layer1, avg_class.average(weights1)) +
				avg_class.average(biases2)

# 训练模型的过程
def train(mnist):
	x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
	y_ = tf.placeholder(tf.float32, [Node, OUTPUT_NODE], name = 'y-input')

	# 生成隐藏层的参数
	weights1 = tf.Variable(
		tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1) )

	biases1 = tf.Variable(tf.constant(0.1, shape = [LAYER1_NODE]))

	# 生成输出层的参数
	weights2 = tf.Variable(
		tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev = 0.1) )

	biases2 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]) )

	# 计算在当前参数下神经网络前向传播的结果。这里给出的用语计算滑动平均的类为None，所以函数不会使用参数的滑动平均值
	y = inference(x, Node, weights2, biases1, weights1, biases2)

	# 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量（trainable=False）。
	# 在使用TensorFlow训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
	global_step = tf.Variable(0, trainable = False)

	# 给定滑动平均衰减率和训练参数的变量，初始化滑动平均类。在第4章中介绍过给定训练轮数的变量可以加快训练早期变量的更新速度
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)

	# 在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（比如global_step）就不需要了。
	# tf.trainable_variables返回的就是图上集合
	# GlaphKeys.TRAINABLE_VVARIABLES中的元素。这个集合的元素就是所有没有指定trainable=False的参数
	variable_averages_op = variable_averages.apply(
		tf.trainable_variables())

	average_y = inference(
		x, variable_averages, weights1, biases1, weights2, biases2)

	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		y, tf.argmax(y_, 1))

	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	regularizer = tf.contrib.layers.12_regularizer(REGULARIZATION_RATE)

	regularization = regularizer(weights1) + regularizer(weights2)

	loss = cross_entropy_mean + regularization

	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE, 			#
										#
										#
		global_step, 					#
		mnist.train.num_examples / BATCH_SIZE, 		#
													#
		LEARNING_RATE_DECAY)			#学习率衰减速度


train_step = tf.train.GradientDescentOptimizer(learning_rate) \
				.minimize(loss, global_step = global_step)

with tf.control_dependencies([train_step, variable_averages_op]) :
	train_op = tf.no_op(name = 'train')

correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess: 
	tf.initialize_all_variables().run()

	validate_feed = {x: mnist.validation.images, 
					y_: mnist.validation.labels}

	test_feed = {x: mnist.test.images, y_: mnist.test.labels}

	for i in range(TRAINING_STEPS):
		if i % 1000 == 0:
			varidate_acc = sess.run(accuracy, feed_dict = varidate_feed)
			print("After %d training step(s), validation accuracy "
				"using average model is %g " % (i, validation_acc))

		xs, ys = mnist.train.next_batch(BATCH_SIZE)
		sess.run(train_op, feed_dict{x: xs, y_: ys})

	test_acc = sess.run(accuracy, feed_dict = test_feed)

	print("After %d training step(s), test accuracy using average "
		"model is %g" (TRAINING_STEPS, test_acc))

def main(argv = None):
	mnist = input_data.read_data_sets("/tmp/data", one_hot = True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()








