import tensorflow as tf
import json
import pandas as pd
import numpy as np
from preprocessing import process_input

def get_data(data_path):
    with open(data_path, 'rb') as f:
        data = json.load(f)
    print 'Loaded {} items'.format(len(data))
    return pd.DataFrame(data)

df = get_data('data/test.json')
X_test, _ = process_input(df)
# bands_1 = np.stack([np.array(band).reshape(75, 75, 1) for band in df['band_1']], axis=0)
# bands_2 = np.stack([np.array(band).reshape(75, 75, 1) for band in df['band_2']], axis=0)
# X_test = np.stack([bands_1, bands_2], axis=3).squeeze()

batch_size = 100

tf.reset_default_graph()
tf.set_random_seed(123)

# Restore variables from disk.
sess = tf.Session()
saver = tf.train.import_meta_graph('./conv1/conv3.meta')
saver.restore(sess, tf.train.latest_checkpoint('./conv1/'))
graph = tf.get_default_graph()

x = graph.get_tensor_by_name("x:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
output = graph.get_tensor_by_name("activ:0")
print "Model restored"
# print sess.run(output, feed_dict={x: X_test[:100], keep_prob: 1.0})
print "Running Predictions"
total_batch = int(len(X_test) / batch_size)
X_batches = np.array_split(X_test, total_batch)
# Loop over all batches
pred = list()
for i in range(total_batch):
    batch_x = X_batches[i]
    result = sess.run(output, feed_dict={x: batch_x, keep_prob: 1.0})
    pred += list(result.squeeze())

df['is_iceberg'] = map(lambda n: round(n, 1), pred)
submission = df[['id', 'is_iceberg']]
submission.to_csv('./submissions/sub3.csv', index=False)


