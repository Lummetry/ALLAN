# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:20:23 2019

@author: damia
"""

from tagger.brain.base_engine import ALLANEngine 
import os
import numpy as np
import tensorflow as tf

from libraries.lummetry_layers_gated import GatedDense

class ALLANTagger(ALLANEngine):
  """
  
  """
  def __init__(self, log,    
               dict_word2index=None,
               dict_label2index=None,
               output_size=None,
               vocab_size=None,
               embed_size=None,
               inputs=None,
               outputs=None,
               columns_end=None,
               DEBUG=False):
    """
    pass either dicts or output_size/vocab_size
    embed_size also optional - will be loaded based on saved embeds
    """
    super().__init__(log=log, DEBUG=DEBUG)
    self.trained = False
    self.__name__ = 'ALLAN_TAG'
    self.output_size = len(dict_label2index) if dict_label2index is not None else output_size
    self.vocab_size = len(dict_word2index) if dict_word2index is not None else vocab_size
    self.dic_word2index = dict_word2index
    if dict_word2index is not None:
      self._get_reverse_word_dict()
      self._get_vocab_stats()
    self.dic_labels = dict_label2index
    self.embed_size = embed_size
    self.pre_inputs = inputs
    self.pre_outputs = outputs
    self.pre_columns_end = columns_end
    self._setup_model()
    return
  
  def _get_vocab_stats(self,):
    if self.dic_word2index is None:
      raise ValueError("Vocab dict is not available !")
    if self.dic_index2word is None:
      raise ValueError("Reverse vocab dict is not available !")
    lens = [len(k) for k in self.dic_word2index]
    dct_stats = {
          "Max" : int(np.max(lens)),
          "Avg" : int(np.mean(lens)),
          "Med" : int(np.median(lens)),
        }
    self.P("Loaded vocab:")
    for k in dct_stats:
      self.P("  {} word size: {}".format(k, dct_stats[k]))
    self.log.ShowTextHistogram(lens)
    return dct_stats
  
  def _setup_word_embeddings(self):
    self.embeddings = None
    fn_emb = self.model_config['EMBED_FILE'] if 'EMBED_FILE' in self.model_config.keys() else ""
    fn_emb = self.log.GetModelFile(fn_emb)
    if os.path.isfile(fn_emb):
      self.P("Loading embeddings {}...".format(fn_emb[-25:]))
      self.embeddings = np.load(fn_emb, allow_pickle=True)
      self.P(" Loaded embeddings: {}".format(self.embeddings.shape))      
    return  
  
  
  def _init_hyperparams(self):
    self.seq_len = self.model_config['SEQ_LEN'] if 'SEQ_LEN' in self.model_config.keys() else None
    if self.seq_len == 0:
      self.seq_len = None
    self.emb_size = self.model_config['EMBED_SIZE'] if 'EMBED_SIZE' in self.model_config.keys() else 0
    if self.embeddings is not None:
      self.emb_size = self.embeddings.shape[-1]
      self.vocab_size = self.embeddings.shape[-2]
    self.emb_trainable = self.model_config['EMBED_TRAIN'] if 'EMBED_TRAIN' in self.model_config.keys() else True
    self.model_columns = self.model_config['COLUMNS']
    
    if self.pre_inputs is not None:
      self.model_input = self.pre_inputs
    else:
      self.model_input = self.model_config['INPUT']
      
    self.use_cuda = self.model_config['USE_CUDA'] if 'USE_CUDA' in self.model_config.keys() else True
    
    if self.pre_outputs:
      self.model_output = self.pre_outputs
    else:
      self.model_output = self.model_config['OUTPUT']
      
    self.dropout_end = self.model_config['DROPOUT_CONCAT'] if 'DROPOUT_CONCAT' in self.model_config.keys() else 0.2 
    self.end_fc = self.model_config['END_FC']    
    return
  
  
  def _define_column(self, tf_input, kernel, filters, name, 
                              activation='relu', last_step='lstm',
                              use_cuda=True,
                              depth=0):
    """
    inputs:
        tf_input: input tensor (batch, seq, features) such as (None, None, 128)
        kernel: the size of the kernel and stride
        filters: size of feature space
        name: unique name of column
    outputs:
        tensor of shape (batch, filters)
    """
    if depth == 0:
      # TODO: smart infer depth of the column!
      depth = max(1, 8 // kernel)
    tf_x = tf_input
    for L in range(1, depth+1):
      tf_x = tf.keras.layers.Conv1D(filters=filters,
                                    kernel_size=kernel,
                                    strides=kernel,
                                    name=name+'_conv{}'.format(L))(tf_x)
      tf_x = tf.keras.layers.BatchNormalization(name=name+'_bn{}'.format(L))(tf_x)
      tf_x = tf.keras.layers.Activation(activation, 
                                        name=name+'_{}{}'.format(activation,L))(tf_x)
    if last_step == 'lstm':
      if use_cuda:
        lyr_last1 = tf.keras.layers.CuDNNLSTM(filters, name=name+'_CUlsmt')
      else:
        lyr_last1 = tf.keras.layers.LSTM(filters, name=name+'_lstm')
      lyr_last2 = tf.keras.layers.Bidirectional(lyr_last1, name=name+'_bidi')
      tf_x = lyr_last2(tf_x)
    elif last_step == 'gp':
      lyr_last1 = tf.keras.layers.GlobalMaxPool1D(name=name+'_GMP')
      lyr_last2 = tf.keras.layers.GlobalAvgPool1D(name=name+'_GAP')
      tf_x1 = lyr_last1(tf_x)
      tf_x2 = lyr_last2(tf_x)
      tf_x = tf.keras.layers.concatenate([tf_x1,tf_x2], name=name+'_concat_gp')
    else:
      raise ValueError("Unknown column summarization method '{}'".format(last_step))
    return tf_x
  
  
  def _get_end_fc(self, tf_x, lst_config_layers):
    for i, layer_config in enumerate(lst_config_layers):
      lyr_name =  layer_config['NAME']
      lyr_type = layer_config['TYPE']
      lyr_units = layer_config['UNITS']
      lyr_act = layer_config['ACTIV']
      lyr_bn = layer_config['BN']
      lyr_drop = layer_config['DROP']
      if lyr_units == 0:
        lyr_units = self.n_concat_outs // (2**(i+1))
      if "gated" in lyr_type.lower(): 
        lyr_gated = GatedDense(units=lyr_units,
                               activation=lyr_act,
                               batch_norm=lyr_bn,
                               name=lyr_name+"_gated{}".format(i+1))
        tf_x = lyr_gated(tf_x)
        tf_x = tf.keras.layers.Dropout(lyr_drop, 
                                       name=lyr_name+'_drop_{}_{}'.format(
                                           lyr_drop,i+1))(tf_x)
      else:
        tf_x = tf.keras.layers.Dense(units=lyr_units,
                                     activation=None,
                                     name=lyr_name+'_dns{}'.format(i+1))(tf_x)
        if lyr_bn:
          tf_x = tf.keras.layers.BatchNormalization(name=lyr_name+'_bn{}'.format(i+1))(tf_x)
        tf_x = tf.keras.layers.Activation(lyr_act, 
                                          name=lyr_name+'_{}{}'.format(lyr_act,i+1))(tf_x)
        tf_x = tf.keras.layers.Dropout(lyr_drop, 
                                       name=lyr_name+'_drop_{}_{}'.format(
                                           lyr_drop,i+1))(tf_x)
    return tf_x
    

  def maybe_load_pretrained(self):
    _res = False
    if "PRETRAINED" in self.model_config:
      fn = self.model_config['PRETRAINED']
      if self.log.GetModelsFile(fn) is not None:
        self.P("Loading pretrained model {}".format(fn))
        self.model = self.log.LoadKerasModel(
                                  fn,
                                  custom_objects={
                                      "GatedDense" : GatedDense,
                                      })
        _res = True
        self._reload_embeds_from_model()
    return _res
      
    
  
  def _setup_model(self):
    self._setup_word_embeddings()
    self._init_hyperparams()
    if self.maybe_load_pretrained():
      self.P("Pretrained model:\n{}".format(
          self.log.GetKerasModelSummary(self.model)))
      self.trained = True
      return
    if 'embeds' in self.model_input.lower():
      tf_input = tf.keras.layers.Input((self.seq_len, self.emb_size), 
                                       name='tagger_input')
      tf_embeds = tf_input
    elif 'tokens' in self.model_input.lower():
      tf_input = tf.keras.layers.Input((self.seq_len,))
      if self.embeddings is not None:
        _init = tf.keras.initializers.Constant(self.embeddings)
      else:
        _init = 'uniform'
  
      lyr_embeds = tf.keras.layers.Embedding(self.vocab_size,
                                             self.emb_size,
                                             embeddings_initializer=_init,
                                             trainable=self.emb_trainable,
                                             name=self.emb_layer_name)
      tf_embeds = lyr_embeds(tf_input)
    else:
      raise ValueError("Uknown model input '{}'".format(self.model_input))
    tf_lst_cols = []
    for i,col in enumerate(self.model_columns):
      n_feats = col['FEATURES']
      ker_size = col['KERNEL']
      col_depth = col['DEPTH']
      end_type = col['END']
      if self.pre_columns_end is not None:
        end_type = self.pre_columns_end
      tf_x = self._define_column(tf_input=tf_embeds,
                                 kernel=ker_size,
                                 name='C'+str(i+1),
                                 filters=n_feats,
                                 depth=col_depth,
                                 last_step=end_type,
                                 use_cuda=self.use_cuda
                                 )
      tf_lst_cols.append(tf_x)
    tf_x = tf.keras.layers.concatenate(tf_lst_cols)
    drp = self.dropout_end 
    tf_x = tf.keras.layers.Dropout(drp, 
                                   name='drop_{}_{}'.format(
                                       drp,0))(tf_x)
    self.n_concat_outs = len(self.model_columns) * n_feats
    
    tf_x = self._get_end_fc(tf_x, self.end_fc)
    
    # now model output
    self.P("Setting model output mode to '{}'".format(self.model_output))
    if 'ranking' in self.model_output:
      ### softmax output
      tf_readout = tf.keras.layers.Dense(self.output_size,
                                         activation='softmax',
                                         name='readout_softmax')(tf_x)
      model = tf.keras.models.Model(inputs=tf_input,
                                    outputs=tf_readout)
      model.compile(optimizer='adam', loss='categorical_crossentropy')
    

    elif self.model_output == 'tagging':
      ### sigmoid output
      tf_readout = tf.keras.layers.Dense(self.output_size,
                                         activation='sigmoid',
                                         name='readout_sigmoid')(tf_x)
      model = tf.keras.models.Model(inputs=tf_input,
                                    outputs=tf_readout)
      model.compile(optimizer='adam', loss='binary_crossentropy')
    else:
      raise ValueError("Unknown model output '{}'".format(self.model_output))
    self.model = model
    self.P("Final model:\n{}".format(self.log.GetKerasModelSummary(self.model)))
    return
  
  
  
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
    if self.unk_words_model_config is None:
      raise ValueError("UNK_WORDS_MODEL not configured - please define dict")
    if len(self.unk_words_model_config['LAYERS']) == 0 :
      raise ValueError("UNK_WORDS_MODEL layers not configured - please define list of layers")
    layers_cfg = self.unk_words_model_config['LAYERS']
    tf_input = tf.keras.layers.Input((None,), name='word_input')
    tf_x = tf_input
    n_layers = len(layers_cfg)
    prev_features = 0
    for L in range(n_layers-1):
      layer_name = layers_cfg[L]['NAME']
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
    layer_name = layers_cfg[-1]['NAME']
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
                            use_cuda=self.use_cuda
                          )
    if 'FINAL_DROP' in self.unk_words_model_config.keys():
      drp = self.unk_words_model_config['FINAL_DROP']
      if drp > 0:
        tf_x = tf.keras.layers.Dropout(drp, name='pre_read_drop{:.1f}'.format(drp))(tf_x)
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
    BATCH_SIZE = self.unk_words_model_batch_size
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
    
    
  
  def train_unk_words_model(self,epochs=20):
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
    steps = self.embeddings.shape[0] // self.unk_words_model_batch_size
    self.unk_words_model.fit_generator(generator=gen, steps_per_epoch=steps, epochs=epochs)
                                       
    # test model on a few unk words
    good_words = ['ochi', 'gura','gat','picior']
    self.P("Testing for {}".format(good_words))
    for word in good_words:
      top = self.get_similar_words(word, top=5)
      self.P(" wrd: '{}' results in: {}".format(word, top))
      
    unk_words = ['capui', 'revedre','spaticul',]
    self.P("Testing for {}".format(unk_words))
    for uword in unk_words:
      top = self.get_unk_word_similar_word(uword, top=5)
      self.P(" unk: '{}' results in: {}".format(uword, top))
    return


        

if __name__ == '__main__':
  from libraries.logger import Logger
  from tagger.brain.data_loader import ALLANDataLoader
  
  cfg1 = "tagger/brain/config_sngl_folder.txt"
  
  use_raw_text = True
  save_model = True
  force_batch = True
  use_model_conversion = False
  epochs = 30
  use_loaded = True
  
  l = Logger(lib_name="ALNT",config_file=cfg1)
  l.SupressTFWarn()
  
  loader = ALLANDataLoader(log=l, multi_label=True, 
                           normalize_labels=False)
  loader.LoadData()
  
  eng = ALLANTagger(log=l, 
                    dict_word2index=loader.dic_word2index,
                    dict_label2index=loader.dic_labels)
  eng.train_unk_words_model()
  if use_raw_text:
    eng.train_on_texts(loader.raw_documents,
                       loader.raw_labels,
                       force_batch=force_batch,
                       n_epochs=epochs,
                       convert_unknown_words=use_model_conversion,
                       save=save_model,
                       skip_if_pretrained=use_loaded)
  else:
    eng.train_on_tokens(loader.x_docs, 
                        loader.y_labels,
                        n_epochs=epochs,
                        force_batch=force_batch,
                        convert_unknown_words=use_model_conversion,
                        save=save_model,
                        skip_if_pretrained=use_loaded)
    
  tags = eng.predict_text("ma doare stomacul")
  l.P("Result::\n {} \n {}".format(tags, ['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))
  
  tags = eng.predict_text("ma doare capul")
  l.P("Result::\n {} \n {}".format(tags, ['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))
  
  tags = eng.predict_text("vreau sa slabesc")
  l.P("Result::\n {} \n {}".format(tags, ['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))
  
  
  #TODO: test various scenarios !!! 

      
    