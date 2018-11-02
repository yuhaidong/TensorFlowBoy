#coding=utf-8

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 两个输入节点。

# 后注：这里为什么使用两个输入节点，这两个输入节点都是什么属性，我还没太搞懂？？？？？？

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')

# 回归问题一般只有一个输出节点
# 后注：注意，y_代表正确结果，y代表预测结果

y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义了一个单层的神经网前向传播的过程，这里就是简单加权和。
w1 = tf.Variable(tf.random_normal([2, 1], stddev = 1, seed = 1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本

loss_less = 10
loss_more = 1

# 后注： 自定义损失函数
loss = tf.reduce_sum(tf.select(tf.greater(y, y_), 
									(y - y_) * loss_more, 
									(y_ - y) * loss_less))

# 后注：定义反向传导算法来优化神经网络中的参数。
# 后注：其实优化的过程就是调整参数的过程，也就是调整w1的过程
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集

# 后注：这里的“1”是伪随机数生成器的种子，其中的每个元素都是[0,1]区间的均匀分布的随机数
rdm = RandomState(1)
dataset_size = 128

# 后注：“dataset_size”是行数，“2”是列数，这里就是要生成128个两列的元素
X = rdm.rand(dataset_size, 2)

# 设置回归的正确值为两个输入的和加上一个随机量。之所以要加上一个随机量是为了加入不可预测的噪音，否则不同损失函数的意义就不大了，
# 因为不同损失函数都会在能完全预测正确的时候最低。一般来说噪音为一个均值为0的小量，所以这里的噪音设置为
# -0.05 ~ 0.05 的随机数。
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

print rdm.rand()

# 训练神经网络
with tf.Session() as sess :
	init_op = tf.initialize_all_variables()
	sess.run(init_op)

	# 后注：设定训练的轮数
	STEPS = 5000

	for i in range(STEPS) :
		# 后注：这里“batch_size”是8，“dataset_size”是128
		start = (i * batch_size) % dataset_size
		end = min(start + batch_size, dataset_size)
		sess.run(train_step, 
					feed_dict = {x : X[start : end], y_ : Y[start : end]} )
		print sess.run(w1)


