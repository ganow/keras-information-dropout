#!/usr/bin/env python
#-*- coding: utf-8 -*-

from argparse import ArgumentParser

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from net import IMG_SIZE_DEFAULT, BETA_DEFAULT, KERNEL_SIZE_DEFAULT, NB_FILTERS_DEFAULT
from utils import prepare_dirs, load_data

NB_EPOCH_DEFAULT = 80
LR_DEFAULT = 0.01
BATCH_SIZE_DEFAULT = 128

p = ArgumentParser(description='Training All-CNN-96 network.')

p.add_argument('datapath', type=str,
               help='/path/to/hdf/data/file')

## model parameters
p.add_argument('--img-size', type=int, nargs=2, default=list(IMG_SIZE_DEFAULT),
               help='pixel size of input image (default: {})'.format(IMG_SIZE_DEFAULT))
p.add_argument('--beta', type=float, default=BETA_DEFAULT,
               help='hyper parameter for KL penalty (default: {})'.format(BETA_DEFAULT))
p.add_argument('--kernel-size', type=int, nargs=2, default=list(KERNEL_SIZE_DEFAULT),
               help='kernel size of Conv2D layer (default: {})'.format(KERNEL_SIZE_DEFAULT))
p.add_argument('--nb-filters', type=int, nargs=4, default=list(NB_FILTERS_DEFAULT),
               help='number of filters for Conv2D blocks (default: {})'.format(NB_FILTERS_DEFAULT))

## optional arguments for reuse model definition and weights
p.add_argument('--model-path', type=str, default=None,
               help='json file defining model architecture (default: None)')
p.add_argument('--weights-path', type=str, default=None,
               help='hdf5 file containing network parameters (default: None)')

## learning parameters
p.add_argument('--lr', type=float, default=LR_DEFAULT,
               help='initial learning rate (default: {})'.format(LR_DEFAULT))
p.add_argument('--nb-epoch', type=int, default=NB_EPOCH_DEFAULT,
               help='number of epochs (default: {})'.format(NB_EPOCH_DEFAULT))
p.add_argument('--batch-size', type=int, default=BATCH_SIZE_DEFAULT,
               help='batch size (default: {})'.format(BATCH_SIZE_DEFAULT))

## data management
p.add_argument('--name', type=str, default=None,
               help='experiment name (default: None)')
p.add_argument('--rootdir', type=str, default='./',
               help='root directory for saving results (default: ./)')
p.add_argument('--pasta-ip', type=str, default=None,
               help='ip address for pastalog (default: None)')

if __name__ == '__main__':
  args = p.parse_args()

  print('... Prepare directories')
  dirs = prepare_dirs(args.rootdir, args.name)

  if args.model_path:
    from keras.models import model_from_json
    from net import load_model
    print('... Load network from: {}'.format(args.model_path))
    model = load_model(args.model_path)
  else:
    from net import get_model
    model = get_model(args.img_size, args.beta, args.kernel_size, args.nb_filters)
    model_definition_file = dirs.modelsdir / 'model.json'
    print('... Save model architecture to: {}'.format(model_definition_file))
    import json
    with model_definition_file.open('w') as f:
      json.dump(json.loads(model.to_json()), f)

  if args.weights_path:
    print('... Load parameters from: {}'.format(args.weights_path))
    model.load_weights(args.weights_path)

  print('... Compile model')
  model.compile(optimizer=Adam(lr=args.lr), loss='categorical_crossentropy',
                metrics=['accuracy'])

  print('... Prepare callbacks')
  callbacks = []
  model_checkpoint = ModelCheckpoint((dirs.checkpointsdir / 'weights_{epoch:03d}_{val_acc:.2f}.h5').as_posix(),
                                     monitor='val_acc', verbose=0,
                                     save_best_only=False, save_weights_only=True,
                                     mode='auto')
  callbacks.append(model_checkpoint)

  csv_logger = CSVLogger((dirs.historydir / 'log.tsv').as_posix(), separator='\t', append=False)
  callbacks.append(csv_logger)

  def schedule(epoch):
    if epoch == 30:
      new_lr = args.lr * 0.1
      print('... Learning rate: {}'.format(new_lr))
      return new_lr
    elif epoch == 60:
      new_lr = args.lr * 0.1 * 0.1
      print('... Learning rate: {}'.format(new_lr))
      return new_lr
    else:
      return args.lr
  lr_scheduler = LearningRateScheduler(schedule)
  callbacks.append(lr_scheduler)

  if args.pasta_ip:
    from pastalog_monitor import PastalogMonitor
    name = args.name if args.name else 'all-cnn-96'
    pastalog_monitor = PastalogMonitor(name=name, root=args.pasta_ip)
    callbacks.append(pastalog_monitor)

  print('... Load data')
  (X_train, Y_train), (X_test, Y_test) = load_data(datapath=args.datapath, img_size=args.img_size)

  print('... Start training')
  model.fit(X_train, Y_train,
            shuffle=True, batch_size=args.batch_size, nb_epoch=args.nb_epoch,
            validation_data=(X_test, Y_test), callbacks=callbacks)
