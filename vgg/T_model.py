import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def dense(x, units):
    w_dense = weight_variable([int(x.get_shape()[1]), units])
    b_dense = bias_variable([units])
    #x @ w_dense + b
    return tf.matmul(x, w_dense)+b_dense

def conv2D(x, ksize, filter, strides=[1,1,1,1], padding='SAME'):
    w_conv = weight_variable([ksize[0], ksize[1], int(x.get_shape()[3]), filter])
    b_conv = bias_variable([filter])
    return tf.nn.conv2d(x, w_conv, strides=strides, padding=padding)+b_conv

def max_pooling(x, ksize=[1,2,2,1]):
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME')

def build_model(x_):

    x = tf.reshape(x_, [-1, 33, 21, 1])

    h_conv1 = tf.nn.relu(conv2D(x      , [3,3], 64))
    h_conv2 = tf.nn.relu(conv2D(h_conv1, [3,3], 64))
    h_pool2 = max_pooling(h_conv2)

    h_conv3 = tf.nn.relu(conv2D(h_pool2, [3,3], 128))
    h_conv4 = tf.nn.relu(conv2D(h_conv3, [3,3], 128))
    h_pool4 = max_pooling(h_conv4)

    h_conv5 = tf.nn.relu(conv2D(h_conv4, [3,3], 256))
    h_conv6 = tf.nn.relu(conv2D(h_conv5, [3,3], 256))
    h_conv7 = tf.nn.relu(conv2D(h_conv6, [3,3], 256))
    h_conv8 = tf.nn.relu(conv2D(h_conv7, [3,3], 256))
    h_pool8 = max_pooling(h_conv8)

    full_shape = h_pool8.get_shape()
    flatten = tf.reshape(h_pool8, [-1, int(full_shape[1])*int(full_shape[2])*int(full_shape[3]) ])

    h_dense1 = tf.nn.relu(dense(flatten, 4096))
    h_dense2 = tf.nn.relu(dense(h_dense1, 4096))
    h_dense3 = tf.nn.relu(dense(h_dense2, 1000))

    logits = dense(h_dense3, 2)
    return logits


if __name__ == '__main__':

    lr = 0.001
    n_batch = 512
    epochs = 200

    x = np.load('data/t_tr_x.npy')
    te_x = np.load('data/t_te_x.npy')
    y = np.load('data/t_tr_y.npy')
    te_y = np.load('data/t_te_y.npy')
    tr_x, vali_x, tr_y, vali_y = train_test_split(x, y, test_size=0.2, random_state=42)


    x_ = tf.placeholder(tf.float32, [None, 33, 21])
    y_true = tf.placeholder(tf.int32, [None])
    y_one_hot = tf.one_hot(y_true, 2)
    logits = build_model(x_)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits))
    optimizer = tf.train.AdamOptimizer(lr)
    train = optimizer.minimize(loss)

    correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(epochs):
        shuffle_index = np.random.permutation(len(tr_x))
        tr_x = tr_x[shuffle_index]
        tr_y = tr_y[shuffle_index]
        for i_batch in range(int(len(tr_x)/n_batch)-1):
            batch_x = tr_x[i_batch*n_batch:(i_batch+1)*n_batch]
            batch_y = tr_y[i_batch*n_batch:(i_batch+1)*n_batch]
            sess.run(train, feed_dict={x_: batch_x, y_true: batch_y})

        print(epoch+1, sess.run(accuracy, feed_dict={x: vali_x, y_true: vali_y}))

    print(sess.run(accuracy, feed_dict={x: te_x, y_true: te_y}))
