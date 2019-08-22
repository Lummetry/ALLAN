import warnings
import numpy as np
import tensorflow as tf

from libraries.logger import Logger
from models_creator.elmo_model import ELMo
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

#remove pesky sklearn warnings...
def warn(*args, **kwargs):
    pass
warnings.warn = warn

warnings.filterwarnings('always')

if __name__ == '__main__':
  logger = Logger(lib_name='RO-ELMo', 
                  config_file='./models_creator/config_elmo.txt',
                  TF_KERAS = True,
                  SHOW_TIME = True)
  logger.SetNicePrints()
  elmo = ELMo(logger,
              data_file_name='rowiki_dialogues_merged_v2',
              word2idx_file='rowiki_dialogues_merged_v2_wordindex_df.csv',
              max_word_length=26)

  elmo.corpus_tokenization()

  elmo.token_sanity_check()

  elmo_model = elmo.build_model()
  
  logger.P('Start training...')
  epochs = 10
  batch_size = 4
  
  elmo.build_batch_list(batch_size)
  training_steps = elmo.number_of_training_batches
  validation_steps = elmo.number_of_validation_batches

  valid_metrics = Metrics(logger, elmo.validation_generator(), validation_steps, batch_size, elmo.idx2word)
  
  elmo_model.fit_generator(elmo.train_generator(), 
                           steps_per_epoch=training_steps, 
                           epochs=epochs,
                           validation_data=elmo.validation_generator(),
                           validation_steps=validation_steps,
                           callbacks=[valid_metrics])
