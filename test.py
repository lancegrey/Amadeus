x = [1,2,3,4,5]
y = {i:i for i in x}
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
print(np.max(x))

x = tf.Variable([[0., 1., 2.]])
d = Dense(10)
print(d.weights)
y = d(x)
init = tf.global_variables_initializer()
s = tf.Session()
s.run(init)
print(s.run(y))

