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
    model = VGG19(weights='imagenet')

    print_layers(model)

    # load and preprocess image
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict the class probabilities
    preds = model.predict(x)

    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('--------------------------------')
    print('Predicted:', decode_predictions(preds, top=3)[0])
    

main()
