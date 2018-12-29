from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 64])
b_fc1 = bias_variable([64])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([64, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = sess.run( accuracy, feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
print(tf.__version__)



np.savetxt("wgt_f/conv1_w.txt", W_conv1.eval(session=sess).reshape(-1,order='F'), delimiter="\n")
np.savetxt("wgt_f/conv1_b.txt", b_conv1.eval(session=sess).reshape(-1,order='F'), delimiter="\n")
np.savetxt("wgt_f/conv2_w.txt", W_conv2.eval(session=sess).reshape(-1,order='F'), delimiter="\n")
np.savetxt("wgt_f/conv2_b.txt", b_conv2.eval(session=sess).reshape(-1,order='F'), delimiter="\n")
np.savetxt("wgt_f/fc1_w.txt", W_fc1.eval(session=sess).reshape(-1,order='F'), delimiter="\n")
np.savetxt("wgt_f/fc1_b.txt", b_fc1.eval(session=sess).reshape(-1,order='F'), delimiter="\n")
np.savetxt("wgt_f/fc2_w.txt", W_fc2.eval(session=sess).reshape(-1,order='F'), delimiter="\n")
np.savetxt("wgt_f/fc2_b.txt", b_fc2.eval(session=sess).reshape(-1,order='F'), delimiter="\n")



print_w_conv1=tf.Print(W_conv1,[W_conv1],"W_conv1:",summarize=1000)
sess.run(print_w_conv1)
print_b_conv1=tf.Print(b_conv1,[b_conv1],"b_conv1:",summarize=1000)
sess.run(print_b_conv1)
print_w_fc2=tf.Print(W_fc2,[W_fc2],"W_fc2:",summarize=1000)
sess.run(print_w_fc2)
print_b_fc2=tf.Print(b_fc2,[b_fc2],"b_fc2:",summarize=1000)
sess.run(print_b_fc2)


#print(" \n\n")
testimg=mnist.train.next_batch(1)

testconv1=tf.reshape(h_conv1,[-1])
testpool1=tf.reshape(h_pool1,[-1])
testconv2=tf.reshape(h_conv2,[-1])
testpool2=tf.reshape(h_pool2,[-1])
testfc1=tf.reshape(h_fc1,[-1])
testfc2=tf.matmul(h_fc1, W_fc2) + b_fc2
testfc2=tf.reshape(testfc2,[-1])
print_testimg=tf.Print(x_image,[x_image],"test_image:",summarize=1000);
print_testconv1=tf.Print(testconv1,[testconv1],"test_conv1:",summarize=100000);
print_testpool1=tf.Print(testpool1,[testpool1],"testpool1:",summarize=100000);
print_testconv2=tf.Print(testconv2,[testconv2],"testconv2:",summarize=100000);
print_testpool2=tf.Print(testpool2,[testpool2],"testpool2:",summarize=100000);
print_testfc1=tf.Print(testfc1,[testfc1],"testfc1:",summarize=100000);
print_testfc2=tf.Print(testfc2,[testfc2],"testfc2:",summarize=100000);
print(" \n\n")
sess.run(print_testimg,feed_dict={x: testimg[0], y_: testimg[1], keep_prob: 1.0})
print(" \n\n")
sess.run(print_testconv1,feed_dict={x: testimg[0], y_: testimg[1], keep_prob: 1.0})
print(" \n\n")
sess.run(print_testpool1,feed_dict={x: testimg[0], y_: testimg[1], keep_prob: 1.0})
print(" \n\n")
sess.run(print_testconv2,feed_dict={x: testimg[0], y_: testimg[1], keep_prob: 1.0})
print(" \n\n")
sess.run(print_testpool2,feed_dict={x: testimg[0], y_: testimg[1], keep_prob: 1.0})
print(" \n\n")
sess.run(print_testfc1,feed_dict={x: testimg[0], y_: testimg[1], keep_prob: 1.0})
print(" \n\n")
sess.run(print_testfc2,feed_dict={x: testimg[0], y_: testimg[1], keep_prob: 1.0})
