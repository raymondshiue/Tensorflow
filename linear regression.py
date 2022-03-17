'''
linear regression using tensorflow v2 with self defined simple dense layer
plot update during model fitting and other info provided on tensorboard
'''

import tensorflow as tf
import tensorboard
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import shutil

dir_path = './logs'

try:
    shutil.rmtree(dir_path)
except OSError as e:
    print("Error: %s : %s" % (dir_path, e.strerror))

class SimpleDense(Layer):
    def __init__(self, units=32, activation=None):
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        with tf.name_scope(self.name):
            with tf.name_scope("weights"):
                w_init = tf.random_normal_initializer()
                self.w = tf.Variable(name="W", initial_value=w_init(shape=(input_shape[-1], self.units),
                        dtype='float32'),trainable=True)
                tf.summary.histogram(self.name +"/weights", self.w)
            with tf.name_scope("biases"):
                b_init = tf.zeros_initializer()
                self.b = tf.Variable(name="b", initial_value=b_init(shape=(self.units,), dtype='float32'), 
                		trainable=True)
                tf.summary.histogram(self.name + "/biases", self.b)
            super().build(input_shape)

    def call(self, inputs):
        with tf.name_scope("Wx_plus_b"):
            outputs=tf.add(tf.matmul(inputs, self.w), self.b)
            tf.summary.histogram(self.name + "/outputs", outputs)
        return self.activation(outputs)

class PerformancePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, lines):
        self.x_test = x_test
        self.lines = lines
        
    def on_epoch_end(self, epoch, logs={}):
    	if epoch %20 ==0:    # update plot every 20 epoch
	    	y_pred = self.model.predict(self.x_test)
	    	try:
	    		ax.lines.remove(self.lines[0])
	    	except Exception:
	    		pass
	    	self.lines = ax.plot(self.x_test,y_pred)
	    	plt.pause(0.1)

xdata = np.linspace(-1,1,300)
noise = np.random.normal(0, 0.05, xdata.shape)
ydata = np.square(xdata) -0.5 +noise

model = tf.keras.models.Sequential([
    SimpleDense(10, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(10, activation='softmax'),
    SimpleDense(1, activation=None)
])

for i in range(len(model.layers)):
    model.layers[i]._name="layer%s" %(i+1)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "logs/", histogram_freq=20)
# tensorboard --logdir logs

ax=plt.subplot(1,1,1)
ax.scatter(xdata,ydata)
lines = ax.plot(xdata,ydata)
plt.ion()
plt.show()
performance_simple=PerformancePlotCallback(xdata, lines)

model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xdata, ydata, epochs=1000, verbose=1, 
        callbacks=[performance_simple, tensorboard_callback])
		#, validation_data=(xdata[:5], ydata[:5]),validation_freq=50)
print(model.evaluate(xdata, ydata))