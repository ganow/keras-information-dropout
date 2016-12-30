#!/usr/bin/env python
#-*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
nb_epoch = 50
# beta = 0.01
beta = 1.0
# lr = 0.001
lr = 0.0001

print('... Define network')

x = Input(batch_shape=(batch_size, original_dim), name='input')
h = Dense(intermediate_dim, activation='relu', name='h')(x)
z = Dense(latent_dim, activation='relu', name='z')(h)
z_logalpha = Dense(latent_dim, name='z_logalpha')(h)


def sampling(args):
  z, z_logalpha = args

  epsilon = K.exp(K.random_normal(shape=K.shape(z), mean=0.,
                                  std=K.exp(z_logalpha)))
  return z * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
noise_z = Lambda(sampling, output_shape=lambda input_shapes: input_shapes[0], name='noise_z')([z, z_logalpha])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu', name='decoder_h')
decoder_mean = Dense(original_dim, activation='sigmoid', name='decoder_x')
h_decoded = decoder_h(noise_z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
  xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
  kl_loss = - K.mean(z_logalpha)
  return xent_loss + beta * kl_loss

vae = Model(x, x_decoded_mean)

print('... Dump model')
print(vae.summary())
with open('data/vae_information_dropout.json', 'w') as f:
  json.dump(json.loads(vae.to_json()), f)

print('... Compile model')
vae.compile(optimizer=Adam(lr=lr), loss=vae_loss)

print('... Load dataset')

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print('... Start training')

model_checkpoint = ModelCheckpoint('data/vae_weights_{epoch:02d}_{val_loss:.2f}.h5',
                                   monitor='val_loss', verbose=0,
                                   save_best_only=False, save_weights_only=False,
                                   mode='auto')

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[model_checkpoint])
