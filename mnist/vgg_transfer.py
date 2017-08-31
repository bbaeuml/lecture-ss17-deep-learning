""" Like logistic_regression_tf_full.py but the subgraph for performing a step of the 
gradient descent optimizer is added using a tensorflow function.

"""

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions

import numpy as np
import math

def print_layers(model):
    print('---------------------------------')
    for l in model.layers:
        print(l.name)
    

def main():
    # Load  the trained model.
    base_model = VGG19(weights='imagenet')

    print_layers(model)

    fc1 = base_model.get_layer('fc1')

    new_fc2 = Dense(1024, activation='relu')(fc1)

    # new prediction layer for our 200 classes
    y =  Dense(200, activation='softmax')(new_fc2)

    # the complete model 
    model = Model(inputs=base_mode.input, output=y)

    # first we only train on the new FC layers and freeze all layers below
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model and use more fance rmsprop optimizer for training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # train the model (fit_generator expects a generator providing the training samples)
    model.fit_generator(...)

    # now we fine-tune the convolution layers of the last block
    layers = ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4']
    for layer in layers:
        layers.trainable = True

    # recompile the model to generate new computational graph for training all un-frozen layers
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # train the last convolutional block of VGG19 and our dense layers
    model.fit_generator(...)
    

main()
