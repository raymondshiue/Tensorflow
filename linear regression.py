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

tf.compat.v1.disable_eager_execution()

class SimpleDense(Layer):
    def __init__(self, units=32, activation=None):
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel", initial_value=w_init(shape=(input_shape[-1], self.units),
                dtype='float32'),trainable=True)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias", initial_value=b_init(shape=(self.units,), dtype='float32'), 
        		trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)

class PerformancePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, lines):
        self.x_test = x_test
        self.lines = lines
        
    def on_epoch_end(self, epoch, logs={}):
    	if epoch%20==0:    # update plot every 20 epoch
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
    # tf.keras.layers.Dense(10, activation='softmax')
    SimpleDense(1, activation=None)
])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "logs/")
# tensorboard --logdir logs

ax=plt.subplot(1,1,1)
ax.scatter(xdata,ydata)
lines = ax.plot(xdata,ydata)
plt.ion()
plt.show()
performance_simple=PerformancePlotCallback(xdata, lines)
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xdata, ydata, epochs=1000,verbose=1,callbacks=[performance_simple, tensorboard_callback])
		#, validation_data=(xdata[:5], ydata[:5]),validation_freq=50)
print(model.evaluate(xdata, ydata))