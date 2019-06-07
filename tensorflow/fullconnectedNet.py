from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets("../../MNIST_data/", one_hot=True)

import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))
x_data = tf.placeholder('float', [None, 784])
y = tf.nn.softmax(tf.matmul(x_data, W) + b)

y_data = tf.placeholder('float', [None, 10])

cross_entropy = -tf.reduce_sum(y_data * tf.log(y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x_data: batch_xs, y_data: batch_ys})

        if step % 200 == 0:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))


#画单张mnist数据集的数据
def drawdigit(position,image,title):
    plt.subplot(*position)
    plt.imshow(image,cmap='gray_r')
    plt.axis('off')
    plt.title(title)

#取一个batch的数据，然后在一张画布上画batch_size个子图
def batchDraw(batch_size):
    images, labels = mnist.train.next_batch(batch_size)
    row_num = math.ceil(batch_size ** 0.5)
    column_num = row_num
    plt.figure(figsize=(row_num,column_num))
    for i in range(row_num):
        for j in range(column_num):
            index = i * column_num + j
            if index < batch_size:
                position = (row_num,column_num,index+1)
                image = images[index].reshape(-1,28)
                title = 'actual:%d'%(np.argmax(labels[index]))
                drawdigit(position,image,title)


batchDraw(200)
plt.show()