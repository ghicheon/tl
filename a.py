import glob
import pickle
import numpy as np
import keras
import cv2
import tensorflow as tf

from keras.models import model_from_yaml

row = 32
col = 32
channel=3

n_classes=2


f = open("y.txt")
yall  = f.readline()
f.close()

y_onehot = [[1,0],[0,1]]


Xtrain = []
ytrain = []

Xtest = []
ytest = []


b = 0

for i,fname in enumerate(range(4000)):
    fname = "data/" + str(fname) + ".jpg"
    
    img = cv2.imread(fname)

    img.astype('float32')
    img =  img/255.0

    if b == 0:
        Xtest.append( cv2.resize(img,(row,col) ) )
        is_red =    int( yall[i] == 'R')
        ytest.append( y_onehot[is_red])
        b = 6
    else:
        Xtrain.append( cv2.resize(img,(row,col) ) )
        is_red =    int( yall[i] == 'R')
        ytrain.append( y_onehot[is_red])
        b = b - 1


X_train, y_train = np.array(Xtrain), np.array(ytrain)
X_test, y_test = np.array(Xtest), np.array(ytest)

print(len(X_train), len(y_train))
print(type(X_train), type(y_train))
print(type(X_train[0]), type(y_train[0]))

#####################################################################################

import tensorflow as tf

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

# Input layer
x  = tf.placeholder(tf.float32, [None, 32*32*3], name='x')
y_ = tf.placeholder(tf.float32, [None, n_classes],  name='y_')
x_image = tf.reshape(x, [-1, 32, 32, 3])

# Convolutional layer 1
W_conv1 = weight_variable([3, 3, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32 , name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, n_classes])
b_fc2 = bias_variable([n_classes])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')


# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver()

nn=0
# Training steps
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  max_steps = 10000
  for step in range(max_steps):
    
    n100  = np.random.choice( len(X_train)-1, 50, replace=False)
    #print(n100)
    Xbatch = X_train[n100].reshape((-1,32*32*3))
    ybatch = y_train[n100]
    
    sess.run(train_step, feed_dict={x: Xbatch, y_:ybatch, keep_prob: 0.5})
    if nn %50 == 0 :
        print("done...  ", step/ float(max_steps))
    nn += 1


    if nn% 9999 == 0:
        saver.save(sess,'./saved_model')



  print( max_steps, sess.run(accuracy, feed_dict={x: X_test.reshape((-1,32*32*3)), 
                                       y_: y_test, keep_prob: 1.0}))
    



