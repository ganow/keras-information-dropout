#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.activations import softmax
from keras import backend as K

from kl_regularizer import KLRegularizer
from conv2d import Convolution2D

IMG_SIZE_DEFAULT = (96, 96)
BETA_DEFAULT = 1.0
KERNEL_SIZE_DEFAULT = (3, 3)
NB_FILTERS_DEFAULT = [32, 64, 96, 192]

if K.image_dim_ordering() == 'tf':
  bn_axis = 3
else:
  bn_axis = 1

def information_dropout_block(input_tensor, kernel_size, nb_filter, beta, block):

  x = Convolution2D(nb_filter, kernel_size[0], kernel_size[1],
                    border_mode='same', name='block{}-conv0'.format(block))(input_tensor)
  x = BatchNormalization(axis=bn_axis, name='block{}-bn0'.format(block))(x)
  x = Activation('relu', name='block{}-relu0'.format(block))(x)

  x = Convolution2D(nb_filter, kernel_size[0], kernel_size[1],
                    border_mode='same', name='block{}-conv1'.format(block))(x)
  x = BatchNormalization(axis=bn_axis, name='block{}-bn1'.format(block))(x)
  x = Activation('relu', name='block{}-relu1'.format(block))(x)

  f_x = Convolution2D(nb_filter, kernel_size[0], kernel_size[1],
                      border_mode='same', name='block{}-conv2'.format(block), subsample=(2, 2))(x)
  f_x = BatchNormalization(axis=bn_axis, name='block{}-bn2'.format(block))(f_x)
  f_x = Activation('relu', name='block{}-relu2'.format(block))(f_x)
  logalpha = Convolution2D(nb_filter, kernel_size[0], kernel_size[1],
                           activity_regularizer=KLRegularizer(beta=beta),
                           border_mode='same', subsample=(2, 2),
                           name='block{}-logalpha'.format(block))(x)

  def sampling(args):
    f_x, logalpha = args

    # clip 0 < alpha < 0.5 for stabilize learning
    epsilon = K.exp(K.random_normal(shape=K.shape(f_x), mean=0.,
                                    std=K.clip(K.exp(logalpha), 0, 0.5)))
    return K.in_train_phase(f_x * epsilon, f_x)

  noise_x = Lambda(sampling, output_shape=lambda input_shapes: input_shapes[0],
                   name='block{}-z'.format(block))([f_x, logalpha])

  return noise_x

def get_model(img_size, beta, kernel_size, nb_filters):

  print('... Define network')
  print('... Parameters:')
  print(' '*8 + 'img_size: {}'.format(img_size))
  print(' '*8 + 'beta: {}'.format(beta))
  print(' '*8 + 'kernel_size: {}'.format(kernel_size))
  print(' '*8 + 'nb_filters: {}'.format(nb_filters))

  img_rows, img_cols = img_size

  if K.image_dim_ordering() == 'th':
    input_shape = (1, img_rows, img_cols)
  else:
    input_shape = (img_rows, img_cols, 1)

  input_tensor = Input(shape=input_shape, name='input')
  x = information_dropout_block(input_tensor, kernel_size, nb_filters[0], beta, block=0)
  x = information_dropout_block(x, kernel_size, nb_filters[1], beta, block=1)
  x = information_dropout_block(x, kernel_size, nb_filters[2], beta, block=2)
  x = information_dropout_block(x, kernel_size, nb_filters[3], beta, block=3)

  x = Convolution2D(192, 3, 3, border_mode='same', name='block4-conv0')(x)
  x = BatchNormalization(axis=bn_axis, name='block4-bn0')(x)
  x = Activation('relu', name='block4-relu0')(x)

  x = Convolution2D(192, 1, 1, border_mode='same', name='block4-conv1')(x)
  x = BatchNormalization(axis=bn_axis, name='block4-bn1')(x)
  x = Activation('relu', name='block4-relu1')(x)

  x = Convolution2D(10, 1, 1, border_mode='same', name='block4-conv2')(x)
  x = BatchNormalization(axis=bn_axis, name='block4-bn2')(x)
  x = Activation('relu', name='block4-relu2')(x)

  x = GlobalAveragePooling2D(name='spatial-average')(x)
  x = Lambda(lambda x: softmax(x), name='softmax')(x)

  model = Model(input_tensor, x, name='All-CNN-96')
  return model

def load_model(json_path, weight_path=None):
  from keras.models import model_from_json
  with open(json_path, 'r') as f:
    model = model_from_json(f.readline(),
                            custom_objects={'Convolution2DwithOwnRegularizer': Convolution2D,
                                            'KLRegularizer': KLRegularizer})
  if weight_path:
    model.load_weights(weight_path)
  return model

if __name__ == '__main__':
  model = get_model(IMG_SIZE_DEFAULT, BETA_DEFAULT, KERNEL_SIZE_DEFAULT, NB_FILTERS_DEFAULT)
  # model = load_model('models/all_cnn_96.json')
  print(model.summary())

  import json
  with open('models/all_cnn_96.json', 'w') as f:
    json.dump(json.loads(model.to_json()), f)
