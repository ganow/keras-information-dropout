#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np

from keras.regularizers import Regularizer
from keras import backend as K
from keras.utils.generic_utils import get_from_module

class KLRegularizer(Regularizer):

  def __init__(self, beta=0.0):
    self.beta = K.cast_to_floatx(beta)

  def __call__(self, logalpha):
    clipped = K.clip(logalpha, np.log(0), np.log(0.5))
    regularization = 0
    regularization += - self.beta * K.mean(K.sum(clipped, keepdims=0))
    return regularization

  def get_config(self):
    return {'name': self.__class__.__name__,
            'beta': float(self.beta)}

def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer',
                           instantiate=True, kwargs=kwargs)
