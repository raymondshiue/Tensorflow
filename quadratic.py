'''
quadratic regression using tensorflow v2 with self defined simple dense layer
plot update during model fitting and other info including graph and 
histograms provided on tensorboard
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
            with tf.name_scope("a"):
                a_init = tf.random_normal_initializer()
                self.a = tf.Variable(name="A", initial_value=a_init(shape=(input_shape[-1], self.units),
                        dtype='float32'),trainable=True)
                tf.summary.histogram(self.name +"/a", self.a)

            with tf.name_scope("b"):
                b_init = tf.random_normal_initializer()
                self.b = tf.Variable(name="B", initial_value=b_init(shape=(input_shape[-1], self.units),
                        dtype='float32'),trainable=True)
                tf.summary.histogram(self.name +"/b", self.b)

            with tf.name_scope("c"):
                c_init = tf.zeros_initializer()
                self.c = tf.Variable(name="C", initial_value=c_init(shape=(self.units,), dtype='float32'), 
                		trainable=True)
                tf.summary.histogram(self.name + "/c", self.c)

            super().build(input_shape)

    def call(self, inputs):
        with tf.name_scope("ax2_bx_c"):
            outputs=tf.matmul(tf.math.square(inputs), self.a) + tf.matmul(inputs, self.b) + self.c
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

	    	self.lines = ax.plot(self.x_test,y_pred, 'r-')
	    	plt.pause(0.1)

xdata = np.linspace(-10,10,300)
noise = np.random.normal(0, 2, xdata.shape)
ydata = np.square(xdata) +5*xdata -3 +noise

''' determine the number of hiddens layers and nodes
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import GridSearchCV
def create_model(layers, activation):
    model =tf.keras.models.Sequential()
    for i,nodes in enumerate(layers):
        model.add(Dense(nodes))
        model.add(Activation(activation))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    return model
model =KerasRegressor(build_fn=create_model,verbose=0)
layers=[[10],[10,10],[100],[100,10],[10,100]]
param_grid=dict(layers=layers,activation=['relu'],batch_size=[32,128],epochs=[100])
grid=GridSearchCV(estimator=model,param_grid=param_grid)
grid_result=grid.fit(xdata,ydata)
print(xdata.size,ydata.size)
print(grid_result.best_params_)
'''

model = tf.keras.models.Sequential([
    SimpleDense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    SimpleDense(100, activation='relu'),
    # tf.keras.layers.Dense(10, activation='softmax')
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

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xdata, ydata, epochs=1000, verbose=1, 
        callbacks=[performance_simple, tensorboard_callback])
		#, validation_data=(xdata[:5], ydata[:5]),validation_freq=50)
print(model.evaluate(xdata, ydata))