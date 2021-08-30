import tensorflow as tf 
import numpy as np
import pandas as pd 

x= np.arange(100*219).reshape([100,219])
print(x.shape)
x.dtype=np.float32
fwd = tf.convert_to_tensor(x)

max_pool = tf.contrib.layers.max_pool2d
avg_pool = tf.contrib.layers.avg_pool2d
dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
weights_initializer = None
weights_regularizer = None
is_training = True
activation_fn = tf.nn.relu
fwd = tf.expand_dims(fwd, -1)
fwd = tf.expand_dims(fwd, -1)
fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                    scope='c1_1')
print('c1_1 shape:',fwd.shape)
fwd = tf.layers.batch_normalization( inputs=fwd, name="bn1_1" ) 
fwd = activation_fn(fwd)
fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                             scope='c1_2')
print('c1_2 shape:',fwd.shape)
fwd = tf.layers.batch_normalization( inputs=fwd, name="bn1_2" )
fwd = activation_fn(fwd)
fwd = max_pool(fwd, [2, 1], scope='p1')  # 14
print('p1 shape:',fwd.shape)

fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 1], stride=2, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                            scope='c2_1')
print('c2_1 shape:',fwd.shape)
fwd = tf.layers.batch_normalization( inputs=fwd, name="bn2_1" ) 
fwd = activation_fn(fwd)
fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 1], stride=1, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                            scope='c2_2')
print('c2_2 shape:',fwd.shape)
fwd = tf.layers.batch_normalization( inputs=fwd,  name="bn_2_2" )
fwd = activation_fn(fwd)
fwd = max_pool(fwd, [2, 1], scope='p2')  # 7
print('p2 shape:',fwd.shape)
fwd = tf.contrib.layers.conv2d(fwd, 70, [3, 1], stride=2, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                            scope='c3_1')
print('c3_1 shape:',fwd.shape)
fwd = tf.layers.batch_normalization( inputs=fwd,  name="bn3_1" ) 
fwd = activation_fn(fwd)
# Dropout should NOT be used right before the graph construction (emb_z). It makes the affinity matrix too sparse. Rather, at least a conv before, such as here for example.
fwd = dropout_f( fwd, rate=0.1, training=is_training, noise_shape=(fwd.get_shape().as_list()[0], 1, 1, fwd.get_shape().as_list()[-1]) ) # ala badGan. Doesnt do much, but to show there is no problem training with dropout here.
fwd = tf.contrib.layers.conv2d(fwd, 80, [3, 1], stride=1,  normalizer_fn = None, normalizer_params = None, activation_fn = None,
                            scope='c3_2')
print('c3_2 shape:',fwd.shape)
fwd = tf.layers.batch_normalization( inputs=fwd,  name="bn3_2" ) 
fwd = activation_fn(fwd)

fwd = avg_pool(fwd, fwd.get_shape().as_list()[1:3], scope='p3')
print('p3 shape:',fwd.shape)
out = tf.layers.flatten(fwd)

init = tf.initialize_all_variables()

with tf.Session() as sess: 
    sess.run(init)
    sess.run(out)