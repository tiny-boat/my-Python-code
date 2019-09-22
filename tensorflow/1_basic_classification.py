#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from warnings import filterwarnings
filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


# ----------------
# 1 Import the Fashion MNIST dataset
# ----------------

# <class 'tensorflow.python.util.deprecation_wrapper.DeprecationWrapper'>
fashion_mnist = keras.datasets.fashion_mnist
# <class 'tuple'>: fashion_mnist.load_data()
# <class 'numpy.ndarray'>: train_images, train_labels……
(train_images, train_labels), (test_images, test_labels) = \
                fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ----------------
# 2 Explore the data
# ----------------

# show first 4 images from the training set with labels
plt.figure(figsize=(10, 10))   # default size is [6.4, 4.8]
for i in range(4):
    '''
    subplot(*args, **kwargs)
        Add a subplot to the current figure.
    '''
    plt.subplot(2, 2, i+1)   # a subplot in i+1 index of 2*2 map
    '''
    xticks(ticks=None, labels=None, **kwargs)
        Get or set the current tick locations and labels of the x-axis.
    '''
    plt.xticks([])   # disable ticks
    plt.yticks([])
    '''
    grid(b=None, which='major', axis='both', **kwargs)
        Configure the grid lines.
    '''
    plt.grid(False)
    '''
    imshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
    alpha=None, vmin=None, vmax=None, origin=None, extent=None,
    shape=<deprecated parameter>, filternorm=1, filterrad=4.0,
    imlim=<deprecated parameter>, resample=None, url=None, *, data=None, **kwargs)
        Display an image, i.e. data on a 2D regular raster.
    '''
    # plt.cm: Builtin colormaps
    plt.imshow(train_images[i], cmap=plt.cm.gist_earth_r)
    '''
    xlabel(xlabel, fontdict=None, labelpad=None, **kwargs)
        Set the label for the x-axis.
    '''
    plt.xlabel(class_names[train_labels[i]])
    '''
    colorbar(mappable=None, cax=None, ax=None, **kw)
        Add a colorbar to a plot.
    '''
    plt.colorbar()

'''
show(*args, **kw)
    Display a figure.
'''
plt.show()


# ----------------
# 3 Preprocess the data
# ----------------

# scale images values from 0~255 to 0~1
train_images, test_images = train_images / 255, test_images / 255


# ----------------
# 4 Build the model
# ----------------

'''
# setup the layers: layer number, layer nodes, activate function
Sequential(layers=None, name=None)
    Linear stack of layers.
    Arguments:
        layers: list of layers to add to the model.
<class 'tensorflow.python.keras.engine.sequential.Sequential'>
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

'''
Activate functions in tf.keras.activiations (activation='funName'):
    softmax
    elu
    selu
    softplus
    softsign
    relu
    tanh
    sigmoid

    hard_sigmoid
    exponential
    linear

Advanced_acitvate functions in tf.python.keras.layers.advanced_activations
    LeakyReLU
    PReLU: https://arxiv.org/abs/1502.01852
    ELU: https://arxiv.org/abs/1511.07289v1
    ThresholdedReLU: https://arxiv.org/abs/1402.3337
    Softmax
    ReLU

Activate functions in tf.nn (activation=tf.nn.funName)
    softmax
    elu
    selu
    softplus
    softsign
    relu
    tanh
    sigmoid

    crelu
    leaky_relu
    log_softmax
    quantized_relu_x
    relu6

'''

'''
compile(self, optimizer, loss=None, metrics=None,
loss_weights=None, sample_weight_mode=None,
weighted_metrics=None, target_tensors=None,
distribute=None, **kwargs)
    Configures the model for training.
'''
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''
optimizers in tf.keras.optimizers
    SGD
    RMSprop: http://www.cs.toronto.edu/~tijmen/csc321/slides
                  /lecture_slides_lec6.pdf
    Adagrad: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    Adadelta: https://arxiv.org/abs/1212.5701
    Adam: https://arxiv.org/abs/1412.6980v8
          https://openreview.net/forum?id=ryQu7f-RZ
    Adamax: https://arxiv.org/abs/1412.6980v8
    Nadam: http://cs229.stanford.edu/proj2015/054_report.pdf;
           http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
'''

'''
losses functions in tf.keras.losses
    mean_squared_error
    mean_absolute_error
    mean_absolute_percentage_error
    mean_squared_logarithmic_error
    squared_hinge
    hinge
    categorical_hinge
    logcosh
    categorical_crossentropy
    sparse_categorical_crossentropy
    binary_crossentropy
    kullback_leibler_divergence
    poisson
    cosine_proximity
'''

'''
metrics in tf.keras.metrics
    binary_accuracy
    categorical_accuracy
    sparse_categorical_accuracy
    top_k_categorical_accuracy
    sparse_top_k_categorical_accuracy
'''


# ----------------
# 5 Train the model
# ----------------

'''
fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
callbacks=None, validation_split=0.0, validation_data=None,
shuffle=True, class_weight=None, sample_weight=None,
initial_epoch=0, steps_per_epoch=None, validation_steps=None,
validation_freq=1, max_queue_size=10, workers=1,
use_multiprocessing=False, **kwargs)
    Trains the model for a fixed number of epochs (iterations on a dataset).
'''
model.fit(train_images, train_labels, epochs=5)


# ----------------
# 6 Evaluate accuracy
# ----------------

'''
evaluate(self, x=None, y=None, batch_size=None, verbose=1,
sample_weight=None, steps=None, callbacks=None, max_queue_size=10,
workers=1, use_multiprocessing=False)
    Returns the loss value & metrics values for the model in test mode.
    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics).
'''
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
'''
overfitting is when a machine learning model performs
worse on new data than on their training data
'''


# ----------------
# 7 Make predictions
# ----------------

'''
predict(self, x, batch_size=None, verbose=0, steps=None,
callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
    Generates output predictions for the input samples.
    Computation is done in batches.
'''

predictions = model.predict(test_images)
predictions[0]   # first predict output
np.argmax(predictions[0])  # first predict label
test_labels[0]   # true label

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    # predicted_label predictions_array*100% (true_label)
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    '''
    bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
        Make a bar plot.
        Return:
           <class 'matplotlib.container.BarContainer'>
    '''
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    '''
    ylim(*args, **kwargs)
        Get or set the y-limits of the current axes.
    '''
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    '''
    thisplot[index]: <class 'matplotlib.patches.Rectangle'>
    set_color is a method inherited from Class Patch:
    '''
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

# Since computation in predict is done in batches, we should expand dims
'''
expand_dims(a, axis)
    Expand the shape of an array.
    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.
'''
img = (np.expand_dims(img, 0))
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)   # angle of rotation is 45 degree
plt.show()

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
