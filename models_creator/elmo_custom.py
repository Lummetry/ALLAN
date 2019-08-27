import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report

class Metrics(Callback):
  """Custom Keras Calback for Multiclass Metrics
     
     This custom callback is a solution for computing multiclass precision, recall and f1-score
     using sklearn, within a Keras training session.
     
     Arguments:
       logger
       validation_generator: generator as used in fit_generator
       validation_steps: number of steps in validation_generator
       batch_size
       idx2word: dictionary to convert the indicies of words when displaying metrics for each class(for visibilty)
  
  """
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
      
    #display validation metrics every 5 epochs  
    if (epoch + 1) % 5 == 0:  
      self.logger.P(classification_report(val_true, val_pred, labels=target_ids, target_names=target_names), noprefix=True)
    
    return
  
class TimestepDropout(tf.keras.layers.Dropout):
    """Timestep Dropout.

    This version performs the same function as Dropout, however it drops
    entire timesteps (e.g., words embeddings in NLP tasks) instead of individual elements (features).

    # Arguments
        rate: float between 0 and 1. Fraction of the timesteps to drop.

    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`

    # Output shape
        Same as input

    # References
        - A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (https://arxiv.org/pdf/1512.05287)
    """

    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape
