#-*- coding: utf-8 -*-

import pathlib
import h5py
from collections import namedtuple

from keras.utils import np_utils
from keras import backend as K

nb_classes = 10

def prepare_dirs(rootdir, name=None):
  print('... Prepare output directories')

  if isinstance(rootdir, str):
    rootdir = pathlib.Path(rootdir)

  Dirs = namedtuple('Dirs',
                    ['modelsdir', 'datadir', 'imagesdir', 'checkpointsdir', 'historydir'])

  if name:
    rootdir /= name

  modelsdir = rootdir / 'models'
  datadir = rootdir / 'data'
  imagesdir = rootdir / 'images'
  checkpointsdir = modelsdir / 'checkpoints'
  historydir = modelsdir / 'history'

  dirs = Dirs(modelsdir=modelsdir, datadir=datadir,
              imagesdir=imagesdir, checkpointsdir=checkpointsdir,
              historydir=historydir)

  for d in dirs:
    if not d.is_dir():
      d.mkdir(parents=True)

  return dirs

def load_data(datapath, img_size):

  img_rows, img_cols = img_size

  if not isinstance(datapath, str):
    datapath = datapath.as_posix()

  with h5py.File(datapath, 'r') as hf:

    X_train = hf.get('X_train').value
    Y_train = hf.get('Y_train').value

    X_test = hf.get('X_test').value
    Y_test = hf.get('Y_test').value

  if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
  else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

  X_train = X_train.astype(K.floatx())
  X_test = X_test.astype(K.floatx())
  X_train /= 255
  X_test /= 255
  print('X_train shape:', X_train.shape)
  print(X_train.shape[0], 'train samples')
  print(X_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  Y_train = np_utils.to_categorical(Y_train, nb_classes)
  Y_test = np_utils.to_categorical(Y_test, nb_classes)

  return (X_train, Y_train), (X_test, Y_test)
