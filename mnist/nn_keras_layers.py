""" Like logistic_regression_tf_full.py but the subgraph for performing a step of the 
gradient descent optimizer is added using a tensorflow function.

"""

from keras.layers import Input, Dense
from keras.models import Model
import keras

from load_mnist import load_mnist
import numpy as np
import math

def one_hot_coding(c, K):
    """computes the one-out-of-K coding for a tensor of shape (N_examples) resulting
 in a tensor of shape (N_examples, K).
    """    
    return np.eye(K)[c]

def main():
    log_dir = '/tmp/mnist/nn_keras_layers'

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
    num_neurons = 30


    # the network layers
    x = Input(shape=(784,))
    h1 = Dense(num_neurons, activation=keras.activations.relu)(x)
    h2 = Dense(num_neurons, activation=keras.activations.relu)(h1)
    h3 = Dense(num_neurons, activation=keras.activations.relu)(h2)
    h4 = Dense(num_neurons, activation=keras.activations.relu)(h3)

    y = Dense(K, activation=keras.activations.softmax)(h4)

    # Define the model and create the computational graph.
    model = Model(inputs=x, outputs=y)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=eta),
                  metrics=[keras.metrics.categorical_accuracy])


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

            # run the model/graph to compute one step of the gradient descent
            [loss, accuracy] = model.train_on_batch(x_batch, t_batch)

        # run the model/graph to compute the accuracy for the test set
        score = model.evaluate(test_x, test_t, verbose=0)

        print("epoch: {0},  test_accuracy: {1}".format(epoch, score[1]))


main()
