""" Like logistic_regression_tf_full.py but the subgraph for performing a step of the 
gradient descent optimizer is added using a tensorflow function.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from load_mnist import load_mnist
import numpy as np


def features(x):
    """computes the (simple linear) feature mapping of a tensor of shape 
    [N_examples, dim_x] by adding a constant feature resulting in 
    a tensor of shape [N_examples, dim_x + 1].
    """
    def helper_fun(z):
        return np.insert(z, 0, 1.)

    return np.apply_along_axis(helper_fun, 1, x)

def activity(W, phi):
    """computes the activity
    Args:
      W: weight matrix of shape (dim_phi, K_classes)
      phi: feature tensor of shape (N_examples, dim_phi)
    Returns:
      Activity tensor of shape (N_examples, K_classes)
    """
    return tf.nn.softmax(tf.matmul(phi, W))

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
    """computes the accuracy of the model with  the weights W (shape (dim_phi, N_classes)) 
for a test set  x (shape (N_examples, dim_x)) and t (shape (N_examples, K_classes)).
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    return  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    

def main():

    train_data, validate_data, test_data  = load_mnist('mnist.pkl.gz')

    # design matrix of shape (num_examples, dim_x); dim_x = 784
    x_all = train_data[0]
    num_examples = x_all.shape[0]

    # label matrix  (N x 1)
    c_all = train_data[1]
    
    # feature mapping phi(x) resulting in shape (num_examples, dim_phi); dim_phi = dim_x + 1
    phi_all = features(x_all)
    dim_phi = phi_all.shape[1]
    
    K = 10 # number of classes
    # target variable  (num_examples, K)
    t_all = one_hot_coding(c_all, K)
    
    # the same for the test data
    test_x = test_data[0]
    test_phi = features(test_x)
    test_c = test_data[1]
    test_t = one_hot_coding(test_c, K)
    

    batch_size = 600
    # learning rate
    eta = 0.13
    max_epochs = 100

    # nodes for input/output of the tensorflow computation graph
    phi = tf.placeholder(tf.float32, [None, dim_phi])
    t = tf.placeholder(tf.float32, [None, K])

    # the weights are now a variable node which the graph can update; initialized with 0
    W = tf.Variable(tf.zeros([dim_phi, K]))

    # define the computation nodes
    y = activity(W, phi)

    cost = cost_fun(y,t)

    # use automatic differentiation to add the nodes to the computation graph for the gradient of the cost 
    dcost = tf.gradients(cost, W)

    # add subgraph to compute gradient descent update step for all variables for
    # cost objective.
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    
    accuracy = accuracy_fun(y, t)

    # open tf session and initialize variables (none in this example)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # training loop
    for epoch in xrange(max_epochs):
        # in each new epoch randomly shuffle the training data
        perm = np.random.permutation(num_examples)
        phi_all = phi_all[perm]
        t_all = t_all[perm]

        # run through the mini batches and update gradient for each
        for end_index in xrange(batch_size, num_examples, batch_size):
            start_index = end_index - batch_size
            phi_batch = phi_all[start_index:end_index]
            t_batch = t_all[start_index:end_index]

            # run the graph to compute one step of the gradient descent
            sess.run([train_step],  feed_dict={phi: phi_batch, t: t_batch})

        # run the graph to compute the accuracy for the test set
        naccuracy = sess.run(accuracy,  feed_dict={phi: test_phi, t: test_t})
        
        print("epoch: {0},  test_accuracy: {1}".format(epoch, naccuracy))

main()
