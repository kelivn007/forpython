import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_point = 1000
vectors_set = []
for i in range(num_point):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

with tf.name_scope('data'):
    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]

# plt.plot(x_data, y_data, 'ro', label='Original data')
# plt.legend()
# plt.show()

with tf.name_scope('parameter'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.random_uniform([1], -1, 1))
        tf.summary.histogram('weight', W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([1]))
        tf.summary.histogram('weight', b)

with tf.name_scope('y_prediction'):
    y = W * x_data + b

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y - y_data))
    tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(0.5)

with tf.name_scope('train'):
    train = optimizer.minimize(loss)

with tf.name_scope('init'):
    init = tf.initialize_all_variables()

sess = tf.Session()

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('logs', sess.graph)

sess.run(init)

for step in range(12):
    sess.run(train)
    rs = sess.run(merged)
    writer.add_summary(rs, step)
    print(step, sess.run(W), sess.run(b))

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.show()
