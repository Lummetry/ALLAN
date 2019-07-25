# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:04:56 2019

@author: Andrei
"""
import numpy as np

import tensorflow as tf

from libraries.generic_obj import LummetryObject
from libraries.lummetry_layers.gated import GatedDense

class EmbeddingApproximator(LummetryObject):
  def __init__(self, log, np_embeds, dct_w2i, dct_i2w, DEBUG=False):
    super().__init__(log=log, DEBUG=DEBUG)
    self.model = None
    self.__name__ = 'EMBA'
    self.trained = False
    return
  
  def startup(self):
    self.emb_gen_model_config = self.config_data['EMB_GEN_MODEL'] if 'EMB_GEN_MODEL' in self.config_data.keys() else None    
    self.emb_gen_model_batch_size = self.emb_gen_model_config['BATCH_SIZE']
    return
    
  
  def get_model(self):
    return self.model
  
  
  def _define_emb_model_layer(self, 
                              tf_inputs, 
                              layer_name,
                              layer_type,
                              layer_features,
                              residual,
                              final_layer,
                              prev_features,
                              use_cuda,
                              ):
    s_name = layer_name.lower()
    s_type = layer_type.lower()
    n_feats = layer_features
    n_prev_feats = prev_features
    b_residual = residual
    sequences = not final_layer
    if (b_residual and final_layer) or (b_residual and 'emb' in s_type):
      raise ValueError("Pre-readound final and post-input embedding layers cannot have residual connection")
    if 'lstm' in s_type:
      if use_cuda:
        cell_lstm = tf.keras.layers.CuDNNLSTM(units=n_feats, 
                                              return_sequences=sequences,
                                              name=s_name+'_culstm')
      else:
        cell_lstm = tf.keras.layers.LSTM(units=n_feats, 
                                         return_sequences=sequences,
                                         name=s_name+'_lstm')
      tf_x = tf.keras.layers.Bidirectional(cell_lstm,
                                           name=s_name+'_bidi')(tf_inputs)
      # double n_feats due to BiDi
      n_feats *= 2
    elif 'emb' in s_type:
      vocab_size = len(self.char_full_voc)
      emb_size = n_feats
      tf_x = tf.keras.layers.Embedding(vocab_size, 
                                       emb_size, 
                                       name=s_name+'_embd')(tf_inputs)
    elif 'conv' in s_type:
      raise ValueError("Conv layers not implemented")
    else:
      raise ValueError("Unknown '' layer type".format(s_type))
    
    if b_residual:
      if n_prev_feats != n_feats:
        tf_x_prev = tf.keras.layers.Dense(n_feats, 
                                          name=s_name+'_trnsf')(tf_inputs)
      else:
        tf_x_prev = tf_inputs
      tf_x = tf.keras.layers.add([tf_x, tf_x_prev],
                                 name=s_name+'_res')
    
    return tf_x
      
  
  def _define_emb_generator_model(self):
    """
    this method defines a simple-n-dirty char level model for 
    embedding approximation of unknown words
    """
    self.P("Constructing unknown words embeddings generator model")
    if self.emb_gen_model_config is None:
      raise ValueError("EMB_GEN_MODEL not configured - please define dict")
    if len(self.emb_gen_model_config['LAYERS']) == 0 :
      raise ValueError("EMB_GEN_MODEL layers not configured - please define list of layers")

    if 'FINAL_DROP' in self.emb_gen_model_config.keys():
      drp = self.emb_gen_model_config['FINAL_DROP']
    else:
      drp = 0

    tf_input = tf.keras.layers.Input((None,), name='word_input')
    
    columns_cfg = self.emb_gen_model_config['COLUMNS']
    lst_columns = []
    for col_name in columns_cfg:
      column_config = columns_cfg[col_name]    
      layers_cfg = column_config['LAYERS']
      tf_x = tf_input
      n_layers = len(layers_cfg)
      prev_features = 0
      for L in range(n_layers-1):
        layer_name = col_name+'_'+layers_cfg[L]['NAME']
        layer_type = layers_cfg[L]['TYPE']
        layer_features = layers_cfg[L]['FEATS']
        residual = layers_cfg[L]['RESIDUAL']      
        tf_x = self._define_emb_model_layer(
                                tf_inputs=tf_x,
                                layer_name=layer_name,
                                layer_type=layer_type,
                                layer_features=layer_features,
                                residual=residual,
                                final_layer=False,
                                prev_features=prev_features,
                                use_cuda=self.use_cuda
                              )
        prev_features = layer_features
      # final pre-readout
      layer_name = col_name+'_'+layers_cfg[-1]['NAME']
      layer_type = layers_cfg[-1]['TYPE']
      layer_features = layers_cfg[-1]['FEATS']
      residual = layers_cfg[-1]['RESIDUAL']      
      tf_x = self._define_emb_model_layer(
                              tf_inputs=tf_x,
                              layer_name=layer_name,
                              layer_type=layer_type,
                              layer_features=layer_features,
                              residual=residual,
                              final_layer=True,
                              prev_features=prev_features,
                              use_cuda=self.use_cuda,
                            )
      lst_columns.append(tf_x)
    
    tf_x = tf.keras.layers.concatenate(lst_columns, name='concat_columns')
    if drp > 0:
      tf_x = tf.keras.layers.Dropout(drp, name='drop1_{:.1f}'.format(drp))(tf_x)
    tf_x = GatedDense(units=self.emb_size*2, name='gated1')(tf_x)
    if drp > 0:
      tf_x = tf.keras.layers.Dropout(drp, name='drop2_{:.1f}'.format(drp))(tf_x)
    tf_readout = tf.keras.layers.Dense(self.emb_size, name='embed_readout')(tf_x)
    model = tf.keras.models.Model(inputs=tf_input, outputs=tf_readout)
    model.compile(optimizer='adam', loss='logcosh')
    self.unk_words_model = model
    self.unk_words_model_trained = False
    self.P("Unknown words embeddings generator model:\n{}".format(
        self.log.GetKerasModelSummary(self.unk_words_model)))
    return


  
  def _convert_embeds_to_training_data(self, min_word_size=5):
    self.P("Converting embeddings to training data...")
    self.P(" Post-processing with min_word_size={}:".format(min_word_size))
    x_data = []
    for i_word in range(self.embeddings.shape[0]):
      if i_word in self.SPECIALS:
        x_data.append([i_word] + [self.PAD_ID]* min_word_size)
      else:
        x_data.append(self.word_to_char_tokens(self.dic_index2word[i_word], 
                                       pad_up_to=min_word_size))
    lens = [len(x) for x in x_data]
    self.log.ShowTextHistogram(lens)
    self._vocab_lens = np.array(lens)
    self._unique_vocab_lens = np.unique(lens)
    self.P(" Training data unique lens: {}".format(self._unique_vocab_lens))
    return np.array(x_data)
          
  def _get_unk_model_generator(self, x_data):  
    BATCH_SIZE = self.emb_gen_model_batch_size
    while True:
      for unique_len in self._unique_vocab_lens:        
        subset_pos = self._vocab_lens == unique_len
        np_x_subset = np.array(x_data[subset_pos].tolist())
        np_y_subset = self.embeddings[subset_pos]
        n_obs = np_x_subset.shape[0]
        n_batches = n_obs // BATCH_SIZE
        for i_batch in range(n_batches):
          b_start = (i_batch * BATCH_SIZE) % n_obs
          b_end = min(n_obs, b_start + BATCH_SIZE)          
          np_x_batch = np_x_subset[b_start:b_end]
          np_y_batch = np_y_subset[b_start:b_end]
          yield np_x_batch, np_y_batch        
    
   
    
  def train_unk_words_model(self,epochs=5):
    """
     trains the unknown words embedding generator based on loaded embeddings
    """
    if self.unk_words_model is None:
      self._define_emb_generator_model()
    min_size = 10
    # get generator
    x_data = self._convert_embeds_to_training_data(
                              min_word_size=min_size)
    gen = self._get_unk_model_generator(x_data)
    # fit model
    n_batches = self.embeddings.shape[0] // self.unk_words_model_batch_size

    losses = []
    avg_losses = []
    for epoch in range(epochs):
      epoch_losses = []
      for i_batch in range(n_batches):
        x_batch, y_batch = next(gen)
        loss = self.unk_words_model.train_on_batch(x_batch, y_batch)
        print("\r Epoch {}: {:>5.1f}% completed [loss: {:.4f}]".format(
            epoch+1, i_batch / n_batches * 100, loss), end='', flush=True)
        losses.append(loss)
        epoch_losses.append(loss)
      print("\r",end="")
      epoch_loss = np.mean(epoch_losses)
      avg_losses.append(epoch_loss)
      self.P("Epoch {} done. loss:{:>7.4f}, avg loss :{:>7.4f}".format(
          epoch+1, epoch_loss,np.mean(avg_losses)))
      self.__debug_unk_words_model(['creerii', 'pumul','capu','galcile'])      
            
    return
  
    
      
  def debug_unk_words_model(self, unk_words):
    self.P("Testing for {}".format(unk_words))
    for uword in unk_words:
      if uword in self.dic_word2index.keys():
        self.P(" 'Unk' word {} found in dict at pos {}".format(uword, self.dic_word2index[uword]))
        continue
      top = self.get_unk_word_similar_word(uword, top=5)
      self.P(" unk: '{}' results in: {}".format(uword, top))
    return
      
      
  def debug_known_words(self, good_words=['ochi', 'gura','gat','picior']):
    self.P("Testing known words {}".format(good_words))
    for word in good_words:
      top = self.get_similar_words(word, top=5)
      self.P(" wrd: '{}' results in: {}".format(word, top))
    return
  
  
if __name__ == '__main__':
  from libraries.logger import Logger
  from tagger.brain.data_loader import ALLANDataLoader
  
  cfg1 = "tagger/brain/config_sngl_folder.txt"
  l = Logger(lib_name="EGEN",config_file=cfg1)
  emb = None
  dct_w = None
  dct_i = None
  
  eng = EmbeddingApproximator(log=l,
                              np_embeds=emb,
                              dct_w2i=dct_w,
                              dct_i2w=dct_i,
                              )
  eng.train_unk_words_model()
  