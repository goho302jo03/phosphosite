import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def dense(x, units):
    w_dense = weight_variable([int(x.get_shape()[1]), units])
    b_dense = bias_variable([units])
    return x @ w_dense + b_dense

def conv2D(x, ksize, filter, strides=[1,1,1,1], padding='SAME'):
    w_conv = weight_variable([ksize[0], ksize[1], int(x.get_shape()[3]), filter])
    b_conv = bias_variable([filter])
    return tf.nn.conv2d(x, w_conv, strides=strides, padding=padding)+b_conv

def max_pooling(x, ksize=[1,2,2,1]):
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME')

def build_model(x_):

    x = tf.reshape(x_, [-1, 33, 21, 1])

    h_conv1 = tf.nn.relu(conv2D(x      , [3,21], 64, strides=[1, 21, 1, 1]))
    h_conv2 = tf.nn.relu(conv2D(h_conv1, [3,1], 64))
    h_pool2 = max_pooling(h_conv2, ksize=[1,1,2,1])

    h_conv3 = tf.nn.relu(conv2D(h_pool2, [3,1], 128))
    h_conv4 = tf.nn.relu(conv2D(h_conv3, [3,1], 128))
    h_pool4 = max_pooling(h_conv4, ksize=[1,1,2,1])

    h_conv5 = tf.nn.relu(conv2D(h_conv4, [3,1], 256))
    h_conv6 = tf.nn.relu(conv2D(h_conv5, [3,1], 256))
    h_conv7 = tf.nn.relu(conv2D(h_conv6, [3,1], 256))
    h_conv8 = tf.nn.relu(conv2D(h_conv7, [3,1], 256))
    h_pool8 = max_pooling(h_conv8, ksize=[1,1,2,1])

    full_shape = h_pool8.get_shape()
    flatten = tf.reshape(h_pool8, [-1, int(full_shape[1])*int(full_shape[2])*int(full_shape[3]) ])

    h_dense1 = tf.nn.relu(dense(flatten, 4096))
    h_dense2 = tf.nn.relu(dense(h_dense1, 4096))
    h_dense3 = tf.nn.relu(dense(h_dense2, 1024))

    logits = dense(h_dense3, 2)
    return logits


if __name__ == '__main__':

    lr = 0.0001
    n_batch = 256
    epochs = 200

    tr_x = np.load('data/t_tr_x.npy')
    vali_x = np.load('data/t_vali_x.npy')
    te_x = np.load('data/t_te_x.npy')

    tr_y = np.load('data/t_tr_y.npy')
    vali_y = np.load('data/t_vali_y.npy')
    te_y = np.load('data/t_te_y.npy')
    # tr_x, vali_x, tr_y, vali_y = train_test_split(x, y, test_size=0.2, random_state=42)


    x_ = tf.placeholder(tf.float32, [None, 33, 21])
    y_true = tf.placeholder(tf.int32, [None])
    y_one_hot = tf.one_hot(y_true, 2)
    logits = build_model(x_)
    pred = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits))
    optimizer = tf.train.AdamOptimizer(lr)
    train = optimizer.minimize(loss)

    correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
    accuracy_sum = tf.reduce_sum(tf.cast(correct_predict, tf.float32))
    auc, update_op = tf.metrics.auc(labels=y_one_hot[:,1], predictions=pred[:,1])

    # init = tf.global_variables_initializer()
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(epochs):

        # train
        shuffle_index = np.random.permutation(len(tr_x))
        tr_x = tr_x[shuffle_index]
        tr_y = tr_y[shuffle_index]
        for i_batch in range(len(tr_x)//n_batch-1):
            batch_x = tr_x[i_batch*n_batch:(i_batch+1)*n_batch]
            batch_y = tr_y[i_batch*n_batch:(i_batch+1)*n_batch]
            sess.run(train, feed_dict={x_: batch_x, y_true: batch_y})

        # validation
        vali_acc_sum = 0
        sess.run(local_init)
        for i_batch in range(len(vali_x)//n_batch-1):
            # batch_x = np.reshape(vali_x[i], [1, 33, 21])
            # batch_y = np.reshape(vali_y[i], [1])
            batch_x = vali_x[i_batch*n_batch:(i_batch+1)*n_batch]
            batch_y = vali_y[i_batch*n_batch:(i_batch+1)*n_batch]
            vali_acc_sum += sess.run(accuracy_sum, feed_dict={x_: batch_x, y_true: batch_y})
            sess.run(update_op, feed_dict={x_: batch_x, y_true: batch_y})

        batch_x = vali_x[(len(vali_x)//n_batch-1)*n_batch:]
        batch_y = vali_y[(len(vali_x)//n_batch-1)*n_batch:]
        vali_acc_sum += sess.run(accuracy_sum, feed_dict={x_: batch_x, y_true: batch_y})
        l, _ = sess.run([loss, update_op], feed_dict={x_: batch_x, y_true: batch_y})

        vali_acc = vali_acc_sum/len(vali_x)
        vali_auc = sess.run(auc)
        print(epoch+1, l, vali_acc, vali_auc)

    # test
    te_acc_sum = 0
    sess.run(local_init)
    for i_batch in range(len(te_x)//n_batch-1):
        # batch_x = np.reshape(te_x[i], [1, 33, 21])
        # batch_y = np.reshape(te_y[i], [1])
        batch_x = te_x[i_batch*n_batch:(i_batch+1)*n_batch]
        batch_y = te_y[i_batch*n_batch:(i_batch+1)*n_batch]
        te_acc += sess.run(accuracy_sum, feed_dict={x_: batch_x, y_true: batch_y})
        sess.run(update_op, feed_dict={x_: batch_x, y_true: batch_y})

    batch_x = te_x[i_batch*n_batch:(i_batch+1)*n_batch]
    batch_y = te_y[i_batch*n_batch:(i_batch+1)*n_batch]
    te_acc_sum += sess.run(accuracy_sum, feed_dict={x_: batch_x, y_true: batch_y})
    sess.run(update_op, feed_dict={x_: batch_x, y_true: batch_y})

    te_acc = te_acc_sum/len(te_x)
    te_auc = sess.run(auc)
    print(te_acc, te_auc)

