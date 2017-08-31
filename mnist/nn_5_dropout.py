""" Like logistic_regression_tf_full.py but the subgraph for performing a step of the 
gradient descent optimizer is added using a tensorflow function.

"""


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
import keras

from load_mnist import load_mnist
import numpy as np
import math


def main():
    log_dir = '/tmp/mnist/nn_keras'

    train_data, validate_data, test_data  = load_mnist('mnist.pkl.gz')

    # design matrix of shape (num_examples, dim_x); dim_x = 784
    x_all = train_data[0]
    num_examples = x_all.shape[0]
    dim_x = x_all.shape[1]
    
    # label matrix  (N x 1)
    c_all = train_data[1]
    
    K = 10 # number of classes
    # target variable  (num_examples, K)
    t_all = keras.utils.to_categorical(c_all)
    
    # the same for the test data
    test_x = test_data[0]
    test_c = test_data[1]
    test_t = keras.utils.to_categorical(test_c, K)
    

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

    h4_dropout = Dropout(0.5)(h4)

    y = Dense(10, activation=keras.activations.softmax)(h4_dropout)

    # Define the model and create the computational graph.
    model = Model(inputs=x, outputs=y)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=eta),
                  metrics=[keras.metrics.categorical_accuracy])

    # Train the model.
    model.fit(x_all, t_all,
              batch_size=batch_size,
              epochs=max_epochs,
              validation_data=(test_x, test_t),                       
              callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10)])

    
    # Evaluate the model
    score = model.evaluate(test_x, test_t, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

main()
