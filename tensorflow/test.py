import tensorflow as tf
import numpy as np

# a = tf.placeholder("float")
# b = tf.placeholder("float")
# y = tf.multiply(a, b)
# sess = tf.Session()
#
# print(sess.run(y, feed_dict={a: 3, b: 3}))


# with tf.name_scope('graph') as scope:
#     matrix1 = tf.constant([[3., 3.]], name='matrix1')
#     matrix2 = tf.constant([[2.], [2.]], name='matrix2')
#     product = tf.matmul(matrix1, matrix2, name='product')
#
# sess = tf.Session()
#
# writer = tf.summary.FileWriter('logs/', sess.graph)
# sess.run(tf.global_variables_initializer())

# print(2 ** 4)

a = np.array([[1,2,3], [4,5,6]])
print(np.reshape(a, (3, 2)))
