#!/usr/bin/env python
#-*- coding: utf-8 -*-

from keras import activations, initializations, constraints
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.utils.np_utils import conv_output_length, conv_input_length

import kl_regularizer as regularizers

class Convolution2DwithOwnRegularizer(Layer):

  def __init__(self, nb_filter, nb_row, nb_col,
               init='glorot_uniform', activation=None, weights=None,
               border_mode='valid', subsample=(1, 1), dim_ordering='default',
               W_regularizer=None, b_regularizer=None, activity_regularizer=None,
               W_constraint=None, b_constraint=None,
               bias=True, **kwargs):
    if dim_ordering == 'default':
      dim_ordering = K.image_dim_ordering()
    if border_mode not in {'valid', 'same', 'full'}:
      raise ValueError('Invalid border mode for Convolution2D:', border_mode)
    self.nb_filter = nb_filter
    self.nb_row = nb_row
    self.nb_col = nb_col
    self.init = initializations.get(init, dim_ordering=dim_ordering)
    self.activation = activations.get(activation)
    self.border_mode = border_mode
    self.subsample = tuple(subsample)
    if dim_ordering not in {'tf', 'th'}:
      raise ValueError('dim_ordering must be in {tf, th}.')
    self.dim_ordering = dim_ordering

    self.W_regularizer = regularizers.get(W_regularizer)
    self.b_regularizer = regularizers.get(b_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)

    self.W_constraint = constraints.get(W_constraint)
    self.b_constraint = constraints.get(b_constraint)

    self.bias = bias
    self.input_spec = [InputSpec(ndim=4)]
    self.initial_weights = weights
    super(Convolution2DwithOwnRegularizer, self).__init__(**kwargs)


  def build(self, input_shape):
    if self.dim_ordering == 'th':
      stack_size = input_shape[1]
      self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
    elif self.dim_ordering == 'tf':
      stack_size = input_shape[3]
      self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
    else:
      raise ValueError('Invalid dim_ordering:', self.dim_ordering)
    self.W = self.add_weight(self.W_shape,
                             initializer=self.init,
                             name='{}_W'.format(self.name),
                             regularizer=self.W_regularizer,
                             constraint=self.W_constraint)
    if self.bias:
      self.b = self.add_weight((self.nb_filter,),
                               initializer='zero',
                               name='{}_b'.format(self.name),
                               regularizer=self.b_regularizer,
                               constraint=self.b_constraint)
    else:
      self.b = None

    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights
    self.built = True

  def get_output_shape_for(self, input_shape):
    if self.dim_ordering == 'th':
      rows = input_shape[2]
      cols = input_shape[3]
    elif self.dim_ordering == 'tf':
      rows = input_shape[1]
      cols = input_shape[2]
    else:
      raise ValueError('Invalid dim_ordering:', self.dim_ordering)

    rows = conv_output_length(rows, self.nb_row,
                              self.border_mode, self.subsample[0])
    cols = conv_output_length(cols, self.nb_col,
                              self.border_mode, self.subsample[1])

    if self.dim_ordering == 'th':
      return (input_shape[0], self.nb_filter, rows, cols)
    elif self.dim_ordering == 'tf':
      return (input_shape[0], rows, cols, self.nb_filter)

  def call(self, x, mask=None):
    output = K.conv2d(x, self.W, strides=self.subsample,
                      border_mode=self.border_mode,
                      dim_ordering=self.dim_ordering,
                      filter_shape=self.W_shape)
    if self.bias:
      if self.dim_ordering == 'th':
        output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
      elif self.dim_ordering == 'tf':
        output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
      else:
        raise ValueError('Invalid dim_ordering:', self.dim_ordering)
    output = self.activation(output)
    return output

  def get_config(self):
    config = {'nb_filter': self.nb_filter,
              'nb_row': self.nb_row,
              'nb_col': self.nb_col,
              'init': self.init.__name__,
              'activation': self.activation.__name__,
              'border_mode': self.border_mode,
              'subsample': self.subsample,
              'dim_ordering': self.dim_ordering,
              'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
              'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
              'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
              'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
              'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
              'bias': self.bias}
    base_config = super(Convolution2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

Convolution2D = Convolution2DwithOwnRegularizer

if __name__ == '__main__':
  from keras.models import Model
  from keras.layers import Input

  img_rows, img_cols = 96, 96
  kernel_size = (3, 3)
  nb_filter = 128

  if K.image_dim_ordering() == 'th':
    input_shape = (1, img_rows, img_cols)
  else:
    input_shape = (img_rows, img_cols, 1)

  input_tensor = Input(shape=input_shape, name='input')
  x = Convolution2D(nb_filter, kernel_size[0], kernel_size[1],
                    border_mode='same')(input_tensor)
  model = Model(input_tensor, x)
  print(model.summary())
