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



sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./saved_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))


# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y_ = graph.get_tensor_by_name("y_:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

#Now, access the op that you want to run. 
y = graph.get_tensor_by_name("y:0")

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
    b = b - 1

X_test, y_test = np.array(Xtest), np.array(ytest)

#out = sess.run(y, feed_dict={x: X_test.reshape((-1,32*32*3))  , 
#                        y_:y_test, keep_prob: 1})
#print(out)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

#############################################################################
#out =  sess.run(accuracy, feed_dict={x: X_test.reshape((-1,32*32*3)), 
#                                        y_: y_test, keep_prob: 1.0})
#############################################################################

#############################################################################
#out =  sess.run(y,    feed_dict={x: X_test.reshape((-1,32*32*3)), 
#                                        y_: y_test, keep_prob: 1.0})
#print(out)
#print(out.shape)
#print(type(out))
#print("len X_test:", len(X_test))
#############################################################################
out =  sess.run(correct_prediction,    feed_dict={x: X_test.reshape((-1,32*32*3)), 
                                        y_: y_test, keep_prob: 1.0})
print(out)

