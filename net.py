#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np

from keras.engine import Layer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras import backend as K

img_rows, img_cols = 96, 96

class InformationDropout(Layer):
  """docstring for InformationDropout"""
  def __init__(self):
    self.supports_masking = True
    super(InformationDropout, self).__init__()

  def build(self, input_shape):
    ## TODO: このフェーズでパラメーターの登録をやる．下みたいなノリで
    # self.gamma = self.add_weight(shape,
    #                                  initializer=self.gamma_init,
    #                                  regularizer=self.gamma_regularizer,
    #                                  name='{}_gamma'.format(self.name))
    pass

  def call(self, x, mask=None):
    ## TODO: このフェーズで入力されるxをどのように変換するかについてを記述する
    # if 0. < self.p < 1.:
    #     noise_shape = self._get_noise_shape(x)
    #     x = K.in_train_phase(K.dropout(x, self.p, noise_shape), x)
    # return x

    return x

  def get_config(self):
    ## TODO: このフェーズでパラメーターを辞書として登録する？
    # config = {'p': self.p}
    config = {}
    base_config = super(InformationDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def get_model():
  ## TODO: dropoutで書いているところをinformation dropoutで置き換える

  kernel_size = (3, 3)
  nb_filters = [32, 64, 96, 192, 10]

  if K.image_dim_ordering() == 'th':
    input_shape = (1, img_rows, img_cols)
  else:
    input_shape = (img_rows, img_cols, 1)

  model = Sequential()

  model.add(Convolution2D(nb_filters[0], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu',
                          input_shape=input_shape))
  model.add(Convolution2D(nb_filters[0], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu'))
  model.add(Convolution2D(nb_filters[0], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu',
                          subsample=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Convolution2D(nb_filters[1], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu'))
  model.add(Convolution2D(nb_filters[1], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu'))
  model.add(Convolution2D(nb_filters[1], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu',
                          subsample=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Convolution2D(nb_filters[2], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu'))
  model.add(Convolution2D(nb_filters[2], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu'))
  model.add(Convolution2D(nb_filters[2], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu',
                          subsample=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Convolution2D(nb_filters[3], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu'))
  model.add(Convolution2D(nb_filters[3], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu'))
  model.add(Convolution2D(nb_filters[3], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu',
                          subsample=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Convolution2D(nb_filters[3], kernel_size[0], kernel_size[1],
                          border_mode='same',
                          activation='relu'))
  model.add(Convolution2D(nb_filters[3], 1, 1,
                          border_mode='same',
                          activation='relu'))
  model.add(Convolution2D(nb_filters[4], 1, 1,
                          border_mode='same',
                          activation='relu'))
  ## TODO: spatial averageというのが何なのかよくわからん
  model.add(Flatten())
  model.add(Activation('softmax'))

  return model

  # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
  #                         border_mode='valid',
  #                         input_shape=input_shape))
  # model.add(Activation('relu'))
  # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
  # model.add(Activation('relu'))
  # model.add(MaxPooling2D(pool_size=pool_size))
  # model.add(Dropout(0.25))

  # model.add(Flatten())
  # model.add(Dense(128))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.5))
  # model.add(Dense(nb_classes))
  # model.add(Activation('softmax'))
  # return model

if __name__ == '__main__':
  # information_dropout = InformationDropout()
  # print(information_dropout.get_config())
  model = get_model()
  print(model.summary())
