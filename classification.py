import tensorflow as tf

mnist = tf.keras.datasets.mnist
(xtrain,ytrain),(xtest,ytest) = mnist.load_data()

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # SimpleDense(10, activation='relu'),
    # tf.keras.layers.Dense(10, activation='softmax')
    tf.keras.layers.Dense(10, activation=None)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=5)
model.evaluate(xtest,  ytest, verbose=2)