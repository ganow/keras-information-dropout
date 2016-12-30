#-*- coding: utf-8 -*-

from keras import callbacks
import pastalog

class PastalogMonitor(callbacks.Callback):
  def __init__(self, name, root='http://localhost:8120', send_rate=5):
    self.root = root
    self.send_rate = send_rate
    self.current_batch = 0
    self.current_epoch = 1
    self.batch_log = pastalog.Log(self.root, '{}_batch'.format(name))
    self.epoch_log = pastalog.Log(self.root, '{}_epoch'.format(name))

  def on_batch_end(self, batch, logs={}):
    self.current_batch += 1
    if self.current_batch % self.send_rate == 0:
      self.batch_log.post('Train loss',
                          value=logs.get('loss').tolist(),
                          step=self.current_batch)
      self.batch_log.post('Train accuracy',
                          value=logs.get('acc').tolist(),
                          step=self.current_batch)

  def on_epoch_end(self, epoch, logs={}):
    self.current_epoch += 1
    self.epoch_log.post('Train loss', value=logs.get('loss').tolist(),
                        step=self.current_batch)
    self.epoch_log.post('Train accuracy', value=logs.get('acc').tolist(),
                        step=self.current_batch)
    self.epoch_log.post('Validation loss',
                        value=logs.get('val_loss').tolist(),
                        step=self.current_batch)
    self.epoch_log.post('Validation accuracy',
                        value=logs.get('val_acc').tolist(),
                        step=self.current_batch)

  def send_custom(self, acc=None, loss=None, val_acc=None, val_loss=None,
                  type='epoch', step=1):
    if type == 'epoch':
      log = self.epoch_log
    else:
      log = self.batch_log

    if acc:
      log.post('Train accuracy', value=acc, step=step)
    if loss:
      log.post('Train loss', value=loss, step=step)
    if val_acc:
      log.post('Validation accuracy', value=val_acc, step=step)
    if val_loss:
      log.post('Validation loss', value=val_loss, step=step)
    return
