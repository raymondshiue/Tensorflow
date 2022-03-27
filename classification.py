import tensorflow as tf

mnist = tf.keras.datasets.mnist
(xtrain,ytrain),(xtest,ytest) = mnist.load_data()

''' determine the number of hiddens layers and nodes
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Flatten, Activation
from sklearn.model_selection import GridSearchCV
def create_model(layers, activation):
    model =tf.keras.models.Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for i,nodes in enumerate(layers):
        model.add(Dense(nodes))
        model.add(Activation(activation))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model =KerasClassifier(build_fn=create_model,verbose=1)
layers=[[10],[32],[10,32],[32,10],[10,10],[100],[100,10],[10,100]]
param_grid=dict(layers=layers,activation=['relu'],batch_size=[32,128],epochs=[10])
grid=GridSearchCV(estimator=model,param_grid=param_grid)
grid_result=grid.fit(xtrain,ytrain)
print(xtrain.size,ytrain.size)
print(grid_result.best_params_)
'''

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # SimpleDense(10, activation='relu'),
    # tf.keras.layers.Dense(10, activation='softmax')
    tf.keras.layers.Dense(10, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=10)
model.evaluate(xtest,  ytest, verbose=2)