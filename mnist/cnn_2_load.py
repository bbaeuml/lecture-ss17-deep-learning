""" Like logistic_regression_tf_full.py but the subgraph for performing a step of the 
gradient descent optimizer is added using a tensorflow function.

"""

from keras.models import Model
import keras

from load_mnist import load_mnist
import numpy as np
import math


def main():
    log_dir = '/tmp/mnist/nn_keras'

    train_data, validate_data, test_data  = load_mnist('mnist.pkl.gz')

    input_width = 28
    K = 10 # number of classes

    # test data
    test_x = np.reshape(test_data[0], (-1, input_width, input_width, 1))
    test_c = test_data[1]
    test_t = keras.utils.to_categorical(test_c, K)


    # Load  the trained model.
    print('-----------------------------')
    print('loading the model ...')
    model = keras.models.load_model('cnn_2_model')

    # Evaluate the model
    print('-----------------------------')
    print('evaluating on the test set')
    score = model.evaluate(test_x, test_t, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

main()
