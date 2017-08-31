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

    
    # design matrix of shape (num_examples, input_width, input_width, channels=1);
    # reshape flattened sample  vector into a samples image
    input_width = 28
    x_all = np.reshape(train_data[0], (-1, input_width, input_width, 1))
    num_examples = x_all.shape[0]
    
    # label matrix  (N x 1)
    c_all = train_data[1]
    
    K = 10 # number of classes
    # target variable  (num_examples, K)
    t_all = keras.utils.to_categorical(c_all)
    
    # the same for the test data
    test_x = np.reshape(test_data[0], (-1, input_width, input_width, 1))
    test_c = test_data[1]
    test_t = keras.utils.to_categorical(test_c, K)
    

    batch_size = 128
    # learning rate
    eta = 0.05
    max_epochs = 200

    # We use ReLU neurons for all but the last layers
    relu = keras.activations.relu

    # the network layers
    x = Input(shape=(input_width, input_width, 1))

    h_c1 = Conv2D(32, (5, 5), activation=relu)(x)
    h_p1 = MaxPooling2D(pool_size=(2,2))(h_c1)

    h_c2 = Conv2D(64, (5, 5), activation=relu)(h_p1)
    h_p2 = MaxPooling2D(pool_size=(2,2))(h_c2)

    h_p2_flat = Flatten()(h_p2)

    h_d1 = Dense(1024, activation=relu)(h_p2_flat)

    h_d1_dropout = Dropout(0.5)(h_d1)

    y = Dense(K, activation=keras.activations.softmax)(h_d1_dropout)

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
              #callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10)]
    )

        # Evaluate the model
    score = model.evaluate(test_x, test_t, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model.
    model.save('cnn_2_model')

main()
