import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocessing import smooth, denoise, color_composite, process_input

def get_data(data_path):
    with open(data_path, 'rb') as f:
        data = json.load(f)
    print 'Loaded {} items'.format(len(data))
    return pd.DataFrame(data)

df = get_data('data/train.json')
# df.inc_angle = pd.to_numeric(df['inc_angle'], errors='coerce')
df.inc_angle = df.inc_angle.replace('na', 0)
df.inc_angle = df.inc_angle.astype(float).fillna(0.0)

X, Y = process_input(df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=123, train_size=0.95)

learning_rate = 0.001
reg_param = 0.001
epochs = 100
batch_size = 50
display_step = 10

tf.reset_default_graph()
tf.set_random_seed(123)
x = tf.placeholder(tf.float32, shape=(None, X.shape[1], X.shape[2], X.shape[3]), name='x')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# def conv_net(x):
# with tf.variable_scope('ConvNet'):
conv1 = tf.layers.conv2d(x,
                         filters=64,
                         kernel_size=[5, 5],
                         padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=123),
                         activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(pool1,
                         filters=128,
                         kernel_size=[3, 3],
                         padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=456),
                         activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
conv3 = tf.layers.conv2d(pool2,
                         filters=256,
                         kernel_size=[3, 3],
                         padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=789),
                         activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2)

pool3_flat = tf.contrib.layers.flatten(pool3)
dropout0 = tf.layers.dropout(pool3_flat, keep_prob)
dense1 = tf.layers.dense(dropout0, units=1024, activation=tf.nn.relu)
dropout1 = tf.layers.dropout(dense1, keep_prob)
dense2 = tf.layers.dense(dropout1, units=256, activation=tf.nn.relu)
dropout2 = tf.layers.dropout(dense2, keep_prob)
dense3 = tf.layers.dense(dropout2, units=64, activation=tf.nn.relu)
# with tf.name_scope("output"):
dropout3 = tf.layers.dropout(dense3, keep_prob)
logits = tf.layers.dense(inputs=dropout3, units=1, name='logits')
# logits = tf.layers.dense(inputs=dropout2, units=1)
# logits = tf.layers.dense(inputs=dropout2, units=2)
activ = tf.sigmoid(logits, name='activ')
# return logits

# preds = conv_net(x)

# with tf.name_scope("loss"):
# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y), name='loss')
# cost = tf.reduce_mean(tf.losses.log_loss(labels=y, predictions=activ, reduction=tf.losses.Reduction.NONE), name='loss')

# cost = tf.losses.log_loss(labels=y, predictions=activ, reduction=tf.losses.Reduction.NONE)
# regularizers = tf.nn.l2_loss(dense2) # + tf.nn.l2_loss(dense3) # + tf.nn.l2_loss(dropout2)
# cost = tf.reduce_mean(cost + reg_param * regularizers)

# with tf.name_scope("train"):
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# with tf.name_scope("accuracy"):
predicted_class = tf.greater(activ, 0.5)
    # with tf.name_scope('correct_prediction'):
correct = tf.equal(predicted_class, tf.equal(y, 1.0))
    # with tf.name_scope('accuracy'):
accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name='accuracy')

# correct = tf.nn.in_top_k(activ, y, 1)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter('./train')
# test_writer = tf.summary.FileWriter('./test')
# writer = tf.train.SummaryWriter('./train', graph=tf.get_default_graph())

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(len(X_train) / batch_size)
        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(Y_train, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            # _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            summary, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.6})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(avg_cost))
            acc_train = accuracy.eval(feed_dict={x: X_train[:100], y: Y_train[:100], keep_prob: 1.0})
            acc_test = accuracy.eval(feed_dict={x: X_test[:100], y: Y_test[:100], keep_prob: 1.0})
            print(epoch, "Train accuracy:", acc_train, "Validation accuracy:", acc_test)
            # print(epoch + 1, "Validation accuracy:", acc_test)
    print("Optimization Finished!")

    saver = tf.train.Saver()
    saver.save(sess, './conv1/conv3')
    print("Model saved")
