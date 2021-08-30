import numpy as np 
import pandas as pd 
import tensorflow as tf


h_dim1 = 150
h_dim2 = 100
h_dim3 = 60
h_dim4 = 30
h_dim5 = 10
h_dim6 = 2

def FC_model(sample_shape = None, num_classes = 2, is_training=False, l2_weight=1e-5, batch_norm_decay=0.99, seed=None):
    # inputs: [Batchsize x features]
    tf_placeholder_x = tf.placeholder("float32", [None] + list(sample_shape), 'discriminator_place_holder_x')
    tf_placeholder_y = tf.placeholder("int32", [None], 'discriminator_place_holder_y')
    fwd = tf_placeholder_x
    tf.summary.scalar('min_int_discriminator_inx', tf.reduce_min(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    tf.summary.scalar('max_int_discriminator_inx', tf.reduce_max(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    
    # weights_initializer = tf.truncated_normal_initializer(stddev=0.01, seed=seed)
    # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, seed=seed, dtype=tf.float32)#An initializer that generates tensors with unit variance.
    weights_regularizer = tf.contrib.layers.l2_regularizer  
    activation_fn = tf.nn.relu  
    USE_BN = batch_norm_decay != -1 # True if not -1. BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    BN_TRAINABLE = True
    # max_pool = tf.contrib.layers.max_pool2d
    # avg_pool = tf.contrib.layers.avg_pool2d
    dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
    # if is_training:
    #     fwd = fwd + tf.random_normal( tf.shape(fwd), mean=0.0, stddev=0.3, seed=seed, dtype=tf.float32 ) #? iGan, badGan, triGan use this and/or dropout
    # layer 1
    fwd = tf.contrib.layers.fully_connected(fwd,h_dim1, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_1_d')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn1_1_d" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    # layer 2
    fwd = tf.contrib.layers.fully_connected(fwd,h_dim2, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_2_d')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training,  name="bn1_2" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    
    # layer 3
    fwd = tf.contrib.layers.fully_connected(fwd,h_dim3, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_3')   
    fwd = tf.layers.batch_normalization(inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn1_3" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    
    # layer 4
    fwd = tf.contrib.layers.fully_connected(fwd,h_dim4, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_4')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn1_4" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = dropout_f( fwd, rate=0.6, seed=seed ) # ala badGan. Doesnt do much, but to show there is no problem training with dropout here.

    # layer 5
    fwd = tf.contrib.layers.fully_connected(fwd, h_dim5, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_5')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn1_5" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    fwd = dropout_f( fwd, rate=0.6, seed=seed )
    # layer 6
    last_logits = tf.contrib.layers.fully_connected(fwd, h_dim6, activation_fn=None, normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_6')    
    predictions = tf.nn.softmax(last_logits)
    # change tf_placeholder_y into onehot 
    ys = tf.one_hot(tf_placeholder_y,num_classes)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=last_logits))
   
    train_step = tf.train.AdamOptimizer(0.0001, beta1=0.9, beta2=0.999, epsilon=1e-07).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(ys,1), tf.argmax(predictions,1)) #argmax 返回一维张量中最大值索引
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 把布尔值转换为浮点型求平均数

    return train_step,tf_placeholder_x,tf_placeholder_y,accuracy,loss

def FC_model2(sample_shape = None, num_classes = 2, is_training=False, l2_weight=1e-5, batch_norm_decay=0.99, seed=None):
    # inputs: [Batchsize x features]
    lr = tf.Variable(0.0001,dtype = tf.float32)
    tf_placeholder_x = tf.placeholder("float32", [None] + list(sample_shape), 'discriminator_place_holder_x')
    tf_placeholder_y = tf.placeholder("int32", [None], 'discriminator_place_holder_y')
    fwd = tf_placeholder_x
    tf.summary.scalar('min_int_discriminator_inx', tf.reduce_min(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    tf.summary.scalar('max_int_discriminator_inx', tf.reduce_max(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    
    # weights_initializer = tf.truncated_normal_initializer(stddev=0.01, seed=seed)
    # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, seed=seed, dtype=tf.float32)#An initializer that generates tensors with unit variance.
    weights_regularizer = tf.contrib.layers.l2_regularizer  
    activation_fn = tf.nn.relu  
    USE_BN = batch_norm_decay != -1 # True if not -1. BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    BN_TRAINABLE = True
    # max_pool = tf.contrib.layers.max_pool2d
    # avg_pool = tf.contrib.layers.avg_pool2d
    dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
    # if is_training:
    #     fwd = fwd + tf.random_normal( tf.shape(fwd), mean=0.0, stddev=0.3, seed=seed, dtype=tf.float32 ) #? iGan, badGan, triGan use this and/or dropout
    # layer 1
    fwd = tf.contrib.layers.fully_connected(fwd,100, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_1_d')   
    fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn1_1" ) if USE_BN else fwd
    fwd = activation_fn(fwd)
    # layer 2
    fwd = tf.contrib.layers.fully_connected(fwd,2, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_2_d')   
    # fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn1_2" ) if USE_BN else fwd
    # fwd = activation_fn(fwd)
    last_logits = fwd
    
    predictions = tf.nn.softmax(last_logits)
    print('name of predictions:{}'.format(predictions.name))
    # change tf_placeholder_y into onehot 
    ys = tf.one_hot(tf_placeholder_y,num_classes)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(ys, last_logits))
    loss = tf.losses.softmax_cross_entropy(onehot_labels = ys, logits = last_logits)
    # train_step = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-07).minimize(loss)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(ys,1), tf.argmax(predictions,1)) #argmax 返回一维张量中最大值索引
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 把布尔值转换为浮点型求平均数

    return train_step,tf_placeholder_x,tf_placeholder_y,accuracy,loss,lr

def FC_model3(sample_shape = None, num_classes = 2, is_training=False, seed=None):
    keep_prob = 0.3
    x = tf.placeholder("float32", [None] + list(sample_shape), 'discriminator_place_holder_x')
    y = tf.placeholder("int32", [None], 'discriminator_place_holder_y')
    ys = tf.one_hot(y,2)
    lr = tf.Variable(0.001,dtype = tf.float32)
    # 创建神经网络
    W1 = tf.Variable(tf.truncated_normal([219,100],stddev=0.1))
    b1 = tf.Variable(tf.zeros([1]))
    # 激活层
    layer1 = tf.nn.relu(tf.matmul(x,W1) + b1)
    # drop层
    layer1 = tf.nn.dropout(layer1,keep_prob=keep_prob)

    # 第二层
    W2 = tf.Variable(tf.truncated_normal([100,50],stddev=0.1))
    b2 = tf.Variable(tf.zeros([1]))
    layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
    layer2 = tf.nn.dropout(layer2,keep_prob=keep_prob)

    # 第三层
    W3 = tf.Variable(tf.truncated_normal([50,2],stddev=0.1))
    b3 = tf.Variable(tf.zeros([1]))
    # prediction = tf.nn.softmax(tf.matmul(layer2,W3) + b3)
    prediction = tf.matmul(layer2,W3) + b3
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))

    # 梯度下降法
    # train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    # train_step = tf.train.AdadeltaOptimizer(lr).minimize(loss)  
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)  
    # train_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss) 
    # train_step = tf.train.AdagradOptimizer(learning_rate=1).minimize(loss) 
    prediction_2 = tf.nn.softmax(prediction)
    

    correct_prediction = tf.equal(tf.argmax(ys,1), tf.argmax(prediction_2,1)) #argmax 返回一维张量中最大值索引
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 把布尔值转换为浮点型求平均数
    return train_step,x,y,accuracy,loss,lr,prediction_2


def CNN_model(sample_shape = None, num_classes = 2, is_training=False, l2_weight=1e-5, batch_norm_decay=0.99, seed=None):
          
    lr = tf.Variable(0.0001,dtype = tf.float32)
    tf_placeholder_x = tf.placeholder("float32", [None] + list(sample_shape), 'discriminator_place_holder_x')
    tf_placeholder_y = tf.placeholder("int32", [None], 'discriminator_place_holder_y')
    # cconv2d has 3 dimensions
    fwd = tf.expand_dims(tf_placeholder_x, -1) # height
    fwd = tf.expand_dims(fwd, -1) # channel
    
    tf.summary.scalar('min_int_discriminator_inx', tf.reduce_min(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    tf.summary.scalar('max_int_discriminator_inx', tf.reduce_max(fwd), collections=[tf.GraphKeys.SUMMARIES] if is_training else ["eval_summaries"])
    
    # weights_initializer = tf.truncated_normal_initializer(stddev=0.01, seed=seed)
    # weights_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, seed=seed, dtype=tf.float32)#An initializer that generates tensors with unit variance.
    weights_regularizer = tf.contrib.layers.l2_regularizer  
    activation_fn = tf.nn.relu  
    USE_BN = batch_norm_decay != -1 # True if not -1. BN puts in tf.GraphKeys.UPDATE_OPS, needs to be put in train_op.
    BN_TRAINABLE = True
    max_pool = tf.contrib.layers.max_pool2d
    avg_pool = tf.contrib.layers.avg_pool2d
    dropout_f = tf.layers.dropout # tf.layers.dropout gets drop_rate. Others get keep_rate.
    with tf.name_scope('cnn_modual'):
        fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                    weights_regularizer=weights_regularizer(l2_weight), scope='c1_1')
        print('c1_1 shape:',fwd.shape)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn1_1" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        fwd = tf.contrib.layers.conv2d(fwd, 32, [3, 1], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                weights_regularizer=weights_regularizer(l2_weight), scope='c1_2')
        print('c1_2 shape:',fwd.shape)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn1_2" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        fwd = max_pool(fwd, [2, 1], scope='p1')  # 14
        print('p1 shape:',fwd.shape)
        fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                weights_regularizer=weights_regularizer(l2_weight), scope='c2_1')
        print('c2_1 shape:',fwd.shape)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn2_1" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        fwd = tf.contrib.layers.conv2d(fwd, 64, [3, 1], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                weights_regularizer=weights_regularizer(l2_weight), scope='c2_2')
        print('c2_2 shape:',fwd.shape)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn_2_2" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        fwd = max_pool(fwd, [2, 1], scope='p2')  # 7
        print('p2 shape:',fwd.shape)
        fwd = tf.contrib.layers.conv2d(fwd, 80, [3, 1], stride=2, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                weights_regularizer=weights_regularizer(l2_weight), scope='c3_1')
        print('c3_1 shape:',fwd.shape)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn3_1" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        # Dropout should NOT be used right before the graph construction (emb_z). It makes the affinity matrix too sparse. Rather, at least a conv before, such as here for example.
        fwd = dropout_f( fwd, rate=0.1, training=is_training, seed=seed ) # ala badGan. Doesnt do much, but to show there is no problem training with dropout here.
        fwd = tf.contrib.layers.conv2d(fwd, 128, [3, 1], stride=1, weights_initializer=weights_initializer, normalizer_fn = None, normalizer_params = None, activation_fn = None,
                                weights_regularizer=weights_regularizer(l2_weight), scope='c3_2')
        print('c3_2 shape:',fwd.shape)
        fwd = tf.layers.batch_normalization( inputs=fwd, momentum=batch_norm_decay, trainable=BN_TRAINABLE, training=is_training, name="bn3_2" ) if USE_BN else fwd
        fwd = activation_fn(fwd)
        
        fwd = avg_pool(fwd, fwd.get_shape().as_list()[1:3], scope='p3')
        print('p3 shape:',fwd.shape)
        fwd = tf.layers.flatten(fwd)
    with tf.name_scope("fc_modual"):
        fwd = tf.contrib.layers.fully_connected(fwd,2, activation_fn=None,normalizer_fn=None,
                                            weights_initializer=weights_initializer,
                                            weights_regularizer=weights_regularizer(l2_weight),
                                            trainable=is_training,
                                            scope='fc_2_d')  
    
    last_logits = fwd      
    predictions = tf.nn.softmax(last_logits)
    print('name of predictions:{}'.format(predictions.name))
    # change tf_placeholder_y into onehot 
    ys = tf.one_hot(tf_placeholder_y,num_classes)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(ys, last_logits))
    loss = tf.losses.softmax_cross_entropy(onehot_labels = ys, logits = last_logits)
    # train_step = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-07).minimize(loss)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(ys,1), tf.argmax(predictions,1)) #argmax 返回一维张量中最大值索引
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 把布尔值转换为浮点型求平均数

    return train_step,tf_placeholder_x,tf_placeholder_y,accuracy,loss,lr,predictions