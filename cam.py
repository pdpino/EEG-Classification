from keras.models import Model
import keras.backend as K
from keras.layers.core import Lambda
import tensorflow as tf
import numpy as np

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def naive_cam(input_model, images):
    conv_output = input_model.layers[2].output 
    cam_extractor = K.function([input_model.input], [conv_output])
    cam = cam_extractor([images])
    cam = cam[0]
    cam = np.maximum(cam, 0)
    cam = np.mean(cam, axis=3)
    cam = np.mean(cam, axis=0)
    heatmap = cam / np.max(cam)
    return cam, heatmap

def grad_cam(input_model, images, category_index):
    nb_classes = 2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    last = Lambda(target_layer, output_shape=target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=last)
    loss = K.sum(model.layers[-1].output)
    conv_output = model.layers[2].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    
    output, grads_val = gradient_function([images])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    return cam, heatmap