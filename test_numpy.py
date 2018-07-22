#coding=utf-8

import tensorflow as tf

# NumPy是一个科学计算的工具包，这里通过NumPy工具包生成模拟数据集，
from numpy.random import RandomState

# 定义训练数据batch的大小

# 后注：P59。
# 后注：反向传导算法实现了一个迭代的过程。在每次迭代的开始，
# 后注：首先需要选取一小部分训练数据，这一小部分数据叫做一个batch。

batch_size = 8

# 定义神经网络的参数，这里是沿用3.4.2小节中给出的神经网络结构

# 后注：第一行的意思是，在TensorFlow中声明一个2*3的矩阵变量，tf.Variable是TensorFlow变量的声明函数，seed是随机种子。
# 后注：TensorFlow中变量的初始值可以设置成随机数、常数或者是通过其他变量的初始值计算得到。
# 后注：下面第一行代码会产生一个2*3的矩阵，矩阵中的元素是均值为0，标准差为2的随机数。tf.random_normal函数
# 后注：可以通过参数mean来指定平均值，在没有制定时默认为0.通过满足正态分布的随机数来初始化神经网络中的参数
# 后注：是一个非常常用的方法。除了正态分布的随机数，TensorFlow还提供了一些其他的随机数生成器。参见P54页表3-2

# 后注：除了使用随机数生成函数来初始化变量，TensorFlow还可以使用常数生成函数初始化变量，比如：tf.Variable(tf.zeros[3])
# 后注：除此之外，TensorFlow还支持通过其他变量的初始值来初始化新的变量，比如：tf.Variable(weights.initialized_value())

# 后注：tf.random_normal结果的默认类型为tf.float32，如果在定义的时候不想使用默认类型，可以dtype来声明变量的具体类型。
# 后注：比如tf.Variable(tf.random_normal([2, 3], dtype=tf.float64, stddev=1, seed=1))将变量类型定义为float64

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的一个维度上使用None可以方便使用不大的batch大小。在训练时需要把数据分
# 成较小的batch，但是在测试时，可以一次性使用全部的数据，当数据集比较小时这样比较
# 方便测试，但数据及比较大时，将大量数据放入一个batch可能会导致内存溢出。

# 后注：如果每轮迭代中选取的数据都要通过常量来表示，那么TensorFlow的计算图将会太大。因为每生成一个常量，TensorFlow都会
# 后注：在计算图中增加一个节点。一般来说，一个神经网络的训练过程会需要经过几百万轮甚至几亿轮的迭代，这样计算图就会非常大，
# 后注：而且效率很低。为了避免这个问题，TensorFlow提供了placeholder机制用于提供输入数据。placeholder相当于定义了
# 后注：一个位置，这个位置中的数据在程序运行时再指定。这样在程序中就不需要生成大量常量来提供输入数据，而只需要将数据
# 后注：通过placeholder传入TensorFlow计算图。在placeholder定义时，这个位置上的数据类型是需要指定的。和其他张量一样，
# 后注：placeholder的类型也是不可以改变的。placeholder中数据的维度信息可以根据提供的数据推导得出，所以不一定要给出。P59

# 后注：定义placeholder作为存放输入数据的地方。这里维度也不一定要定义。但如果维度是确定的，那么给出维度可以降低
# 后注：出错的概率。

# 后注：placeholder的作用就是用来输入数据（特征向量）的。之前简单的使用过
# 后注：x = tf.constant([0.7, 0.9])
# 后注：来输入数据。但是用每次都是用常量来输入数据（特征向量）对系统消耗太大了，所以就是用placeholder来定义一个位置，
# 后注：这个位置中的数据在程序运行时才执行。

# 后注：-----------------训练的过程-----------------
# 后注：训练的过程其实就是说，一方面通过让输入值，也就是x的值，经过与w1，w2交互计算之后，得到y的值，也就是预算结果；而另一方面，直接从x拿到正确结果，也就是y_，
# 后注：最后比较y与y_的差异，也就是通过损失函数计算出y与y_的差异，然后输入到反向传导算法中去，优化w1和w2的值。

x = tf.placeholder(tf.float32, shape=(None, 2), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name = 'y-input')

# 定义神经网络前向传播的过程

# 后注：函数tf.matmul实现了矩阵乘法的功能。
# 后注：“x”是一开始输入的值，也就是特征向量
# 后注：“w1”是第1层节点的参数，就是特征向量与第1层节点之间各个边上的权重矩阵
# 后注：“w2”是第2层节点的参数，就是特征向量与第2层节点之间各个边上的权重矩阵

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法。

# 后注：定义损失函数来刻画预测值与真实值的差异。P61
# 后注：tf.reduce_mean()函数是求平均值

cross_entropy = -tf.reduce_mean(
					y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# 后注：定义反向传导算法来优化神经网络中的参数。P61
# 后注：其实优化的过程就是调整参数的过程，也就是调整w1，w2的过程

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集

# 后注：这里的“1”是伪随机数生成器的种子，其中的每个元素都是[0,1]区间的均匀分布的随机数
rdm = RandomState(1)
dataset_size = 128


# 后注：生成一个128行，2列的矩阵，类似于：
'''
array([[ 0.4173048 ,  0.55868983],
       [ 0.14038694,  0.19810149],
       [ 0.80074457,  0.96826158],
       [ 0.31342418,  0.69232262],
       [ 0.87638915,  0.89460666],
       [ 0.08504421,  0.03905478]])
'''

X = rdm.rand(dataset_size, 2)

# 定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为是正样本（比如零件合格），
# 而其他为负样本（比如零件不合格）。和TensorFlow游乐场中的表示法不大一样的地方是，
# 在这里使用0来表示负样本，1来表示正样本。大部分解决分类问题的神经网络都会采用
# 0和1的表示方法。

# 后注：其实这里Y矩阵中的数据就是X矩阵数据（特征向量）对应的正确结果，简单的以两个值的和是否大于1来判断

# 后注：生成一个128行，1列的矩阵，类似于：
'''
[	[1], 
	[1], 
	[0], 
	[1]，
	......
]
'''

Y = [[int(x1 + x2 < 1)] for (x1, x2) in X] 

# 创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
	# 初始化变量。
	
	# 后注：虽然之前在变量定义时已经对变量进行了声明和赋值，但其实这些操作是没有执行的
	# 后注：所以这里需要专门去进行初始化操作。单个变量的初始化操作其实可以通过w1.initialize函数来进行
	# 后注：但当变量较多时那样比较麻烦。
	# 后注：TensorFlow中有一个简便的方法，通过函数tf.initialize_all_variables()函数初始化所有变量。
	
	init_op = tf.initialize_all_variables()

	sess.run(init_op)

	# 后注：输出训练前w1，w2这两个矩阵的值
	
	print sess.run(w1)
	print sess.run(w2)

	'''
	在训练之前神经网络参数的值
	w1 = [[-0.81131822, 1.48459876, 0.06532937]
		[-2.44270396, 0.0992484, 0.59122431]]
	w2 = [[-0.81131822, 1.48459876, 0.06532937]]
	'''

	# 设定训练的轮数。
	STEPS = 5000
	# 后注：从0到4999
	for i in range(STEPS):
		# 每次选取batch_size个样本进行训练。
		start = (i * batch_size) % dataset_size
		end = min(start + batch_size, dataset_size)
	
		# 通过选取的样本训练神经网络并更新参数
		
		# 后注：train_step是前面定义的反向传导算法
		# 后注：前面通过tf.placeholder定义了变量x，y_的位置，这里要 输入它们的值了，就是通过feed_dict来输入的
		# 后注：注意，“[]”是python中的切片方法。
		
		sess.run(train_step, 
					feed_dict={x: X[start : end], y_ : Y[start : end]})
				
		if i % 1000 == 0:
			# 每隔一段时间计算在所有数据上的交叉熵并输出。
			
			# 后注：cross_entropy是前面定义的损失函数
			# 后注：注意这里的x和y_，都是所有数据，而上面的知识从start到end的一个batch里面的数据
			
			total_cross_entropy = sess.run(
				cross_entropy, feed_dict = {x : X, y_ : Y})

			'''
			print ('----------------------------')
			print total_cross_entropy
			print ('----------------------------')
			'''
		
			print("After %d training step(s), cross entropy on all data is %g" %(i, total_cross_entropy))
		
			'''
			输出结果：
			After 0 training step(s), cross entropy on all data is 0.0674925
			After 1000 training step(s), cross entropy on all data is 0.0163385
			After 2000 training step(s), cross entropy on all data is 0.00907547
			After 3000 training step(s), cross entropy on all data is 0.00714436
			After 4000 training step(s), cross entropy on all data is 0.00578471
			
			通过这个结果可以发现随着训练的进行，交叉熵是逐渐变小的。交叉熵越小说明
			预测的结果和真实的结果差距越小。
			'''
			
	# 后注：输出训练后w1，w2这两个矩阵的值
	
	print sess.run(w1)
	print sess.run(w2)

	'''
	在训练之后神经网络参数的值：
	w1 = [[-1.9618273, 2.58235407, 1.68203783]
			[-3.4681716, 1.06982327, 2.11788988]]
	w2 = [[-1.8247149], [2.68546653], [1.41819501]]
	
	可以发现这两个参数的取值已经发生了变化，这个变化就是训练的结果。
	这使得这个神经网络能更好的拟合提供的训练数据。
	'''







