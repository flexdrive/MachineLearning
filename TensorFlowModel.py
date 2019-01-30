#from tensorflow.examples.tutorials.mnist import input_data """This is inputing the tutorial examples"""
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf #imports tensorflow
import numpy as np
#import bag_of_words_gen as data
import json
from bag_of_words_gen import DataIngestion as DI
import logging

log = logging.getLogger('skynet')
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

np.set_printoptions(threshold='nan')

corpModel = DI()
#6 Decisions

historical_data, new_data = corpModel.test_model_ingestion("incidents_corpsys", "newIncidents", 6) #equivalent to mnist.test.images #equivalent to mnist.test.labels"""

historical_words = historical_data['words']
historical_labels = historical_data['labels']
new_words = new_data['words']
new_labels = new_data['labels']


# print(ticket_data.shape)
# print(actionable_data.shape)

historical_ind = list(range(0,len(historical_words)))
new_ind = list(range(0,len(new_words)))

ticket_data_test = np.take(new_words,new_ind, axis=0)
ticket_data_train = np.take(historical_words,historical_ind,axis=0)

actionable_data_test = np.take(new_labels,new_ind,axis=0)
actionable_data_train = np.take(historical_labels,historical_ind, axis=0)

#print(ticket_data_test.shape)
#print(ticket_data_train.shape)
# print("ACTIONABLE: ")
# print(actionable_data_test)
# print("ACTIONABLE: ")
# print(ticket_data_test)


num_keywords = ticket_data_train.shape[1] #should be 625 - defines features, i.e. dimensions in vector space (d)
num_tickets = ticket_data_train.shape[0] #should 6542 - defines the number of points (n)

log.info("num_keywords {} and num_tickets {}".format(num_keywords, num_tickets))

x = tf.placeholder(tf.float32, [None, num_keywords]) #creates a placeholder that can be fed using 'feed_dict' in Session.run().
#must have size of dimensions. Since each data point is in R^d. Each time a session is run, then

W = tf.Variable(tf.zeros([num_keywords, 6])) #weights for linear classifier. Stars it as an array of 0's
b = tf.Variable(tf.zeros([6])) #empty array that will be filled
#y=Wx+b

y = tf.nn.softmax(tf.matmul(x, W) + b) #tensorflow matrix multiplcation: y = W.x + b -> linear classifier
#generalized version of the logisitic function. Input is the linear classifier function of x.W + b
#i.e. y = 1/(1+e^(-x.W + b))


y_ = tf.placeholder(tf.float32, [None, 6]) #private variable y_ - will be the placeholder for test data as the modle minimizes the cross_entropy


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy) #cross_entropy loss run


sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for i in range(50000):  #1000 runs
  batch_xs, batch_ys = ticket_data_train, actionable_data_train
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # classification = tf.run(y, feed_dict={x: batch_xs})
  # print(classification)
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  if i % 100 == 0:
    train_accuracy_first = accuracy.eval(feed_dict={
        x: batch_xs, y_: batch_ys})
    log.info('step %d, training accuracy %r' % (i, train_accuracy_first))

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
log.info(sess.run(accuracy, feed_dict={x: ticket_data_test, y_: actionable_data_test})) #printing probability



#
# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=0.1)
#   return tf.Variable(initial)
#
# def bias_variable(shape):
#   initial = tf.constant(0.1, shape=shape)
#   return tf.Variable(initial)
#
#
# def conv2d(x, W):
#   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
# def max_pool_2x2(x):
#   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')
#
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
#
# #print(W_conv1.shape)
# #print(b_conv1.shape)
# print("x.shape: ")
# print(x.shape)
# print("W_conv1 shape: ")
# print(W_conv1.shape)
# print("b_conv1 shape: ")
# print(b_conv1.shape)
#
# x_image = tf.reshape(x, [-1, 25, 25, 1])
#
# #print(x_image.shape)
#
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# #print(h_conv1.shape)
# #print(h_pool1.shape)
#
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# #print(W_conv2.shape)
# #print(b_conv2.shape)
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# #print(h_conv2.shape)
# #print(h_pool2.shape)
#
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# #print(W_fc1.shape)
# #print(b_fc1.shape)
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)\
#
# #print(h_pool2_flat.shape)
# #print(h_fc1.shape)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# #print(keep_prob.shape)
# #print(h_fc1_drop.shape)
#
# W_fc2 = weight_variable([1024, 6])
# b_fc2 = bias_variable([6])
#
# #print(W_fc2.shape)
# #print(b_fc2.shape)
#
# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#
# #print(y_conv.shape)
#
#
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
#
# #print(cross_entropy.shape)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# #print(correct_prediction.shape)
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   for i in range(200):
#     batch_x, batch_y = ticket_data_train, actionable_data_train
#     if i % 10 == 0:
#       train_accuracy = accuracy.eval(feed_dict={
#           x: batch_x, y_: batch_y, keep_prob: 1.0})
#       print('step %d, training accuracy %g' % (i, train_accuracy))
#     train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
#
#   print('test accuracy %g' % accuracy.eval(feed_dict={
#       x: ticket_data_test, y_: actionable_data_test, keep_prob: 1.0}))
