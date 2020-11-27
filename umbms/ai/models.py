"""
Tyson Reimer
University of Manitoba
January 31st, 2020
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import tensorflow.keras as keras

###############################################################################


def get_sino_cnn(input_shape, lr=0.001):
    """Gets a CNN network using the keras Sequential() object

    input_shape : array_like
        The expected shape of the input data during training/testing,
        ex: for samples that are 35x72, set to [35, 72, 1]
    use_bn : bool
        If True, uses batch normalization after each layer (and does
        *not* use dropout)
    be_deep : bool
        If True, uses the deep CNN network. If False, uses the shallow CNN
    lr : float
        The learning rate to be used with the optimizer
    use_adam : bool
        If True, uses the Adam optimizer. If False, uses stochastic
        gradient descent

    Returns
    -------
    network : keras.models.Sequential()
        The CNN computation graph, already compiled, ready for training.
    """

    # Define the network and add the first layer
    network = Sequential()

    network.add(Conv2D(filters=8,
                       kernel_size=[3, 3],
                       strides=1,
                       padding='same',
                       input_shape=input_shape))
    network.add(Activation('relu'))

    # Add a pooling layer and flatten before final dense layers
    network.add(MaxPooling2D(pool_size=[7, 1], strides=1))

    network.add(Flatten())

    # Add final output layer with softmax activation (i.e., this is the
    # classification layer)
    network.add(Dense(2))
    network.add(Activation('softmax'))

    # Get an optimizer for the network
    optimizer_fn = keras.optimizers.Adam(lr=lr)

    # Compile the network, make it ready for training
    # Use categorical_crossentropy loss for one-hot binary classification
    network.compile(loss='categorical_crossentropy',
                    optimizer=optimizer_fn,
                    metrics=[keras.metrics.AUC(name='auc'),])

    return network


def get_sino_dnn(input_shape, lr=0.001):
    """Gets a DNN network using the keras Sequential() object

    input_shape : array_like
        The expected shape of the input data during training/testing,
        ex: for samples that are 35x72, set to [35, 72, 1]
    use_bn : bool
        If True, uses batch normalization after each layer (and does
        *not* use dropout)
    be_deep : bool
        If True, uses the deep CNN network. If False, uses the shallow CNN
    lr : float
        The learning rate to be used with the optimizer
    use_adam : bool
        If True, uses the Adam optimizer. If False, uses stochastic
        gradient descent

    Returns
    -------
    network : keras.models.Sequential()
        The DNN computation graph, already compiled, ready for training.
    """

    # Define the network and add the first layer
    network = Sequential()

    network.add(Flatten(input_shape=input_shape))  # Flatten input

    network.add(Dense(16))
    network.add(Activation('relu'))

    network.add(Dense(16))
    network.add(Activation('relu'))

    # Add final output layer with softmax activation (i.e., this is the
    # classification layer)
    network.add(Dense(2))
    network.add(Activation('softmax'))

    # Get an optimizer for the network
    optimizer_fn = keras.optimizers.Adam(lr=lr)

    # Compile the network, make it ready for training
    # Use categorical_crossentropy loss for one-hot binary classification
    network.compile(loss='categorical_crossentropy',
                    optimizer=optimizer_fn,
                    metrics=[keras.metrics.AUC(name='auc'),])

    return network
