""" Like logistic_regression_tf_full.py but the subgraph for performing a step of the 
gradient descent optimizer is added using a tensorflow function.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from load_mnist import load_mnist
import numpy as np
import math

def cost_fun(y, t):
    """computes the cost from the predicted probabilities y target variables t. 
 Both tensors are of shape (N_examples, K_classes).
    """
    return tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))

def one_hot_coding(c, K):
    """computes the one-out-of-K coding for a tensor of shape (N_examples) resulting
 in a tensor of shape (N_examples, K).
    """    
    return np.eye(K)[c]

def accuracy_fun(y, t):
    """computes the accuracy of the model with  the weights W (shape (dim_x, N_classes)) 
for a test set  x (shape (N_examples, dim_x)) and t (shape (N_examples, K_classes)).
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    return  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    

def main():
    log_dir = '/tmp/mnist/nn_5_summaries'

    train_data, validate_data, test_data  = load_mnist('mnist.pkl.gz')

    # design matrix of shape (num_examples, dim_x); dim_x = 784
    x_all = train_data[0]
    num_examples = x_all.shape[0]
    dim_x = x_all.shape[1]
    
    # label matrix  (N x 1)
    c_all = train_data[1]
    
    K = 10 # number of classes
    # target variable  (num_examples, K)
    t_all = one_hot_coding(c_all, K)
    
    # the same for the test data
    test_x = test_data[0]
    test_c = test_data[1]
    test_t = one_hot_coding(test_c, K)
    

    batch_size = 600
    # learning rate
    eta = 0.01
    max_epochs = 1000

    sess = tf.InteractiveSession()
    
    # nodes for input/output of the tensorflow computation graph
    x = tf.placeholder(tf.float32, [None, dim_x], name='x')
    t = tf.placeholder(tf.float32, [None, K], name='target')


    # define the layers
    num_neurons = [30, 30, 30, 30]

    num_neurons_1 = num_neurons[0]
    with tf.name_scope('layer_1'):
        W1 = tf.Variable(tf.truncated_normal([dim_x, num_neurons_1], stddev=1.0 / math.sqrt(float(dim_x))), name='weights')
        b1 = tf.Variable(tf.zeros([num_neurons_1]), name='bias')

        h1 = tf.sigmoid(tf.matmul(x, W1) + b1)

    num_neurons_2 = num_neurons[1]
    with tf.name_scope('layer_2'):
        W2 = tf.Variable(tf.truncated_normal([num_neurons_1, num_neurons_2], stddev=1.0 / math.sqrt(float(num_neurons_1))), name='weights')
        b2 = tf.Variable(tf.zeros([num_neurons_2]), name='bias')

        h2 = tf.sigmoid(tf.matmul(h1, W2) + b2)

    num_neurons_3 = num_neurons[2]
    with tf.name_scope('layer_3'):
        W3 = tf.Variable(tf.truncated_normal([num_neurons_2, num_neurons_3], stddev=1.0 / math.sqrt(float(num_neurons_2))), name='weights')
        b3 = tf.Variable(tf.zeros([num_neurons_3]), name='bias')

        h3 = tf.sigmoid(tf.matmul(h2, W3) + b3)

    num_neurons_4 = num_neurons[3]
    with tf.name_scope('layer_4'):
        W4 = tf.Variable(tf.truncated_normal([num_neurons_3, num_neurons_4], stddev=1.0 / math.sqrt(float(num_neurons_4))), name='weights')
        b4 = tf.Variable(tf.zeros([num_neurons_4]), name='bias')

        h4 = tf.sigmoid(tf.matmul(h3, W4) + b4)
        
        
    with tf.name_scope('layer_5'):
        W5 = tf.Variable(tf.truncated_normal([num_neurons_4, K], stddev=1.0 / math.sqrt(float(num_neurons_4))), name='weights')
        b5 = tf.Variable(tf.zeros([K]), name='bias')

        h5 = tf.matmul(h4, W5) + b5

        
    # cost function
    with tf.name_scope('cost'):
        y = tf.nn.softmax(h5)
        cost = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))

    # add subgraph to compute gradient descent update step for all variables for
    # cost objective.
    with tf.name_scope('train_step'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)


    with tf.name_scope('accuracy'):
        accuracy = accuracy_fun(y, t)

    cost_summary = tf.summary.scalar('cost', cost)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
    
    # open tf session and initialize variables (none in this example)
    tf.global_variables_initializer().run()

    # training loop
    for epoch in xrange(max_epochs):
        # in each new epoch randomly shuffle the training data
        perm = np.random.permutation(num_examples)
        x_all = x_all[perm]
        t_all = t_all[perm]

        # run through the mini batches and update gradient for each
        for end_index in xrange(batch_size, num_examples, batch_size):
            start_index = end_index - batch_size
            x_batch = x_all[start_index:end_index]
            t_batch = t_all[start_index:end_index]

            # run the graph to compute one step of the gradient descent
            [_, summary] = sess.run([train_step, cost_summary],  feed_dict={x: x_batch, t: t_batch})

        train_writer.add_summary(summary, epoch)

        # run the graph to compute the accuracy for the test set
        [naccuracy, summary] = sess.run([accuracy, accuracy_summary],  feed_dict={x: test_x, t: test_t})
        train_writer.add_summary(summary, global_step=epoch)

        print("epoch: {0},  test_accuracy: {1}".format(epoch, naccuracy))

    train_writer.close()
main()
