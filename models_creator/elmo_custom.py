import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report 
from tensorflow.keras.callbacks import Callback

class Metrics(Callback):
  def __init__(self, logger, validation_generator, validation_steps, batch_size, idx2word):
    super().__init__()
    self.logger = logger
    self.validation_generator = validation_generator
    self.validation_steps = validation_steps
    self.batch_size = batch_size
    self.idx2word = idx2word
    
    self.train_preds = []
    self.train_true = []
    self.var_y_true = tf.Variable(0., validate_shape=False)
    self.var_y_pred = tf.Variable(0., validate_shape=False)

  def on_train_begin(self, logs):
    self.val_f1_micros = []
    self.val_f1_macros = []

    self.val_recalls = []
    self.val_precisions = []

  def _get_validation_preds(self):
    val_true = []
    val_pred = []
    
    val_true = np.array(val_true)
    val_pred = np.array(val_pred)
    
    for batch in range(self.validation_steps):
      xVal, yVal = next(self.validation_generator)

      val_true = np.hstack((val_true, np.squeeze(np.asarray(yVal), axis=-1).ravel()))
      val_pred = np.hstack((val_pred, np.argmax(np.asarray(self.model.predict(xVal)).round(), axis=-1).ravel()))

    val_true = np.asarray(val_true, dtype=np.int32)
    val_pred = np.asarray(val_pred, dtype=np.int32)
    
    return val_true, val_pred
    
  def on_epoch_end(self, epoch, logs):
          
    val_true, val_pred = self._get_validation_preds()
    
    target_ids = np.unique(val_true)
    target_names = []
    for i in target_ids:
      target_names.append(self.idx2word[i])
      
    self.logger.P(classification_report(val_true, val_pred, labels=target_ids, target_names=target_names), noprefix=True)
    
    return
