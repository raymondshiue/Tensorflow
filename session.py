'''
linear regression using tensorflow v1 and session
'''

import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

xdata = np.random.rand(100).astype(np.float32)
ydata = xdata*0.3 + 0.5

grad = tf.Variable(tf.random.uniform([1],-1,1))
bias = tf.Variable(tf.zeros([1]))

def cost():
	y = xdata*grad + bias

	loss = tf.reduce_mean(tf.square(y-ydata))
	return loss

optimizer = tf.optimizers.SGD(0.5)	# learning rate
train = optimizer.minimize(cost, [grad, bias])

init = tf.compat.v1.initialize_all_variables()
sess = tf.compat.v1.Session()
sess.run(init)

for step in range(201):
	sess.run(train)
	if step%20==0:
		print(step,sess.run(grad),sess.run(bias))