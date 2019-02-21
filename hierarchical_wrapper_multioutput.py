from tqdm import trange
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, concatenate, TimeDistributed
from tensorflow.keras.layers import Bidirectional, CuDNNLSTM, Dense, Embedding, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

from metrics import compute_bleu
from sklearn.metrics import accuracy_score
import pandas as pd
import keras.backend as K
import random


valid_lstms = ['unidirectional', 'bidirectional']
str_optimizers = ['rmsprop', 'sgd', 'adam', 'nadam', 'adagrad']
str_losses = ['mse', 'mae', 'sparse_categorical_crossentropy'] #TODO sparse_.... in TF

__VER__ = "1.1.0"

"""
@history:
  
  2019-02-08: uploaded on github. Current functionalities:
    - hierarchical encoder - decoder
    - character embeddings
    - intent prediction
    - insertion of the last encoder intent as an embedding in the decoder
    
  2019-02-12:
    - integrated bot intent prediction and peeking
    - integrated validation methodology

"""


class HierarchicalNet:
  def __init__(self, logger, data_processer):
    self._version = __VER__
    
    self.logger = logger
    self.data_processer = data_processer
    self.config_data = self.logger.config_data
    self.has_decoder = False
    self._parse_config_data()

    self.tf_graph = None
    self.trainable_model = None
    
    self.is_timedistributed = False
    
    self.enc_pred_model = None
    self.enc_tf_inputs = None
    self.enc_tf_out = None
    
    self.dec_pred_model = None
    self.dec_tf_rec_input = None
    self.dec_tf_inputs = None
    self.dec_tf_out = None
    self.decoder_readout = None

    self.epoch_loaded_model = 0
    
    self._initialize_datastructures()
    
    self._log("Initialized HierarchicalNet v{}".format(self._version))
    
    return
  
  def _initialize_datastructures(self):
    self.input_tensors = {}
    self.EmbLayers = {}
    self.enc_layers_full_state = OrderedDict()
    self.enc_full_state = []
    self.dec_lstm_cells = []
    return
  
  
  def _parse_config_data(self):
    self.model_trained_layers = []
    self.encoder_architecture = self.config_data["ENCODER_ARCHITECTURE"]
    self.decoder_architecture = {}
    
    self._use_keras       = bool(self.config_data['USE_KERAS'])
    self._metrics_config  = self.config_data['METRICS']
    self._str_optimizer   = self.config_data['OPTIMIZER'].lower()
    self._learning_rate   = self.config_data['LEARNING_RATE']
    self._str_loss        = self.config_data['LOSS'].lower()
    self._model_name      = self.config_data['MODEL_NAME']

    self._model_name = self.logger.file_prefix + '_' + self._model_name
    assert self._str_optimizer in str_optimizers
    assert self._str_loss in str_losses
    
    if 'DECODER_ARCHITECTURE' in self.config_data:
      self.decoder_architecture = self.config_data["DECODER_ARCHITECTURE"]

    self.model_trained_layers = self.__get_trained_layers_names()   
    
    self.max_words = self.config_data["MAX_WORDS"]
    self.max_characters = self.config_data["MAX_CHARACTERS"]
    self.nr_user_labels = self.config_data["NR_USER_LABELS"]
    self.nr_bot_labels  = self.config_data["NR_BOT_LABELS"]

    return

  def __get_trained_layers_names(self):
    names = []
     
    if self.encoder_architecture['CHILD1'] != {}:
      names.append('time_distributed')
 
    for d in self.encoder_architecture['PARENT']['LAYERS']:
      names.append(d['NAME'] + '_' + str(d['NR_UNITS']))
   
    names.append('user_intent_dense1')
    names.append('user_intent_logits')
    if 'BOT_INTENT' in self.encoder_architecture['PARENT']:
      if bool(self.encoder_architecture['PARENT']['BOT_INTENT']):
        names.append('bot_intent_dense1')
        names.append('bot_intent_logits')
    
    if 'EMBEDDINGS' in self.encoder_architecture['PARENT']:
      for d in self.encoder_architecture['PARENT']['EMBEDDINGS']:
        if 'TRAINABLE' in d:
          if bool(d['TRAINABLE']) is True:
            names.append(d['NAME'])
    
 
    if 'READOUT' in self.encoder_architecture['PARENT']:
      names.append('readout_' + str(self.encoder_architecture['PARENT']['READOUT']['ACTIVATION']))
 
    if self.decoder_architecture != {}:
      for d in self.decoder_architecture['LAYERS']:
        names.append(d['NAME'] + '_' + str(d['NR_UNITS']))
      if 'EMBEDDINGS' in self.decoder_architecture:
        for d in self.decoder_architecture['EMBEDDINGS']:
          if 'TRAINABLE' in d:
            if bool(d['TRAINABLE']) is True:
              names.append(d['NAME'])
      if 'READOUT' in self.decoder_architecture:
        names.append('readout_' + str(self.decoder_architecture['READOUT']['ACTIVATION']))
          
    return names


  def _log(self, str_msg, results = False, show_time = False, noprefix=False):
    self.logger.VerboseLog(str_msg, results=results, show_time=show_time,
                           noprefix=noprefix)
    return
  
  def _get_last_key_val_odict(self, odict):
    for k in odict.keys():
      continue
    return k, odict[k]


  def _get_key_index_odict(self, odict, key):
    for i,k in enumerate(odict.keys()):
      if k == key:
        return i
    return

  class EmbeddingLayer:
    def __init__(self, output_shape, input_dim=None, output_dim=None,
                 weights=None, name=None, trainable=True):
      self.input_dim    = input_dim
      self.output_dim   = output_dim
      self.output_shape = output_shape
      self.weights      = weights
      self.name         = name
      self.trainable    = trainable
      self.build()
      return

    def build(self):
      with tf.name_scope(self.name):
        if self.weights is None:
          assert self.input_dim is not None
          assert self.output_dim is not None
          self.kernel = tf.Variable(tf.random_uniform(shape=[self.input_dim, self.output_dim],
                                                      name='rand_emb'),
                                    name='embeddings_matrix',
                                    dtype=tf.float32,
                                    trainable=self.trainable)
        else:
          self.kernel = tf.Variable(self.weights, name='embeddings_matrix', dtype=tf.float32,
                                    trainable = self.trainable)

    def __call__(self, inputs):
      return Lambda(lambda i: tf.nn.embedding_lookup(self.kernel, i, name='emb_lookup'),
                    output_shape=self.output_shape,
                    name=self.name)(inputs)

    
  def _is_input_for_embedding(self, input_name, config_embeddings):
    for emb in config_embeddings:
      if input_name == emb['CONNECTED_TO']: return True
    return False


  def _convert_shape(self, shape):
    new_shape = []
    for dim in shape:
      if dim < -1: dim = None
      new_shape.append(dim)
    return tuple(new_shape)

  def CreateChildNet(self, configurations):

    tf_input = Input(shape=(self.max_words + self.max_characters, ), name='input_words_characters')

    def _slice(x, start, end):
      return x[:, start:end]

    start, end = 0, self.max_words
    tf_input_words = Lambda(_slice, arguments={'start': start, 'end': end})(tf_input)
    start, end = self.max_words, self.max_words + self.max_characters
    tf_input_chars = Lambda(_slice, arguments={'start': start, 'end': end})(tf_input)

    config_emb_words = configurations[0]['EMBEDDINGS'][0]
    config_emb_chars = configurations[1]['EMBEDDINGS'][0]

    input_dim, output_dim, weights, trainable, name = None, None, None, True, None
    identifier = configurations[0]['IDENTIFIER']
    if 'NAME' in config_emb_words: name = config_emb_words['NAME']
    if 'INPUT_DIM' in config_emb_words: input_dim = config_emb_words['INPUT_DIM']
    if 'OUTPUT_DIM' in config_emb_words: output_dim = config_emb_words['OUTPUT_DIM']
    if 'TRAINABLE' in config_emb_words: trainable = config_emb_words['TRAINABLE']
    if 'EMB_MATRIX_PATH' in config_emb_words:
      if config_emb_words['EMB_MATRIX_PATH'] != '':
        path = config_emb_words['EMB_MATRIX_PATH']
        if config_emb_words['USE_DRIVE']:
          path = self.logger.GetDataFile(path)
      
        weights = np.load(path)
        assert len(weights.shape) == 2
        if 'PAD' in config_emb_words:
          nr_paddings = config_emb_words['PAD']
          if nr_paddings:
            n_cols = weights.shape[1]
            weights = np.concatenate((weights, np.zeros((1, n_cols))))
            weights = np.concatenate((weights, np.random.uniform(low=-0.05, high=0.05,
                                                                 size=(nr_paddings-1, n_cols))))
            input_dim += nr_paddings
    with tf.name_scope(identifier):
      if weights is None:
        Emb = Embedding(input_dim=input_dim, output_dim=output_dim, trainable=trainable,
                        name=name)
      else:
        Emb = Embedding(input_dim=input_dim, output_dim=output_dim, weights=[weights],
                        trainable=trainable, name=name)
      
      self.EmbLayers[name] = Emb
      tf_emb_words = Emb(tf_input_words)
    
    input_dim, output_dim, weights, trainable, name = None, None, None, True, None
    identifier = configurations[1]['IDENTIFIER']
    if 'NAME' in config_emb_chars: name = config_emb_chars['NAME']
    if 'INPUT_DIM' in config_emb_chars: input_dim = config_emb_chars['INPUT_DIM']
    if 'OUTPUT_DIM' in config_emb_chars: output_dim = config_emb_chars['OUTPUT_DIM']
    if 'TRAINABLE' in config_emb_chars: trainable = config_emb_chars['TRAINABLE']

    with tf.name_scope(identifier):
      Emb = Embedding(input_dim=input_dim, output_dim=output_dim, trainable=trainable,
                      name=name)

      self.EmbLayers[name] = Emb
      tf_emb_chars = Emb(tf_input_chars)
    

    tf_out1 = self.CreateRecurrentCell(configurations[0], tf_emb_words)
    tf_out2 = self.CreateRecurrentCell(configurations[1], tf_emb_chars)

    tf_concat = concatenate([tf_out1, tf_out2], name='word_char_level')
    tf_out = tf_concat

    self.func_child = Model(inputs=tf_input, outputs=[tf_out])

    self._log("Child model\n{}".format(self.logger.GetKerasModelSummary(self.func_child)))

    return self.func_child

  def CreateParentNet(self, configuration, func_child=None):
    tf_input, crt_inputs = self.CreateInputsAndEmbeddings(configuration, func_child)
    self.enc_tf_inputs = list(crt_inputs.values())
    identifier = configuration['IDENTIFIER']

    if self._use_keras:
      assert self.is_timedistributed

    additional_inputs = []
    if func_child is not None and not self.is_timedistributed:
      tf_child_out = func_child.output
      with tf.name_scope(identifier):
        if len(tf_child_out.get_shape()) == 2:
          tf_child_out = tf.expand_dims(tf_child_out, axis=0)
        #endif
      #endwith
      additional_inputs = func_child.inputs

      tf_input = concatenate([tf_input, tf_child_out])
    #endif

    tf_out = self.CreateRecurrentCell(configuration, tf_input, save_state=True)

    if 'READOUT' in configuration:
      tf_out = self.DenseNet(configuration['READOUT'], prefix='readout')(tf_out)

    self.enc_tf_out = tf_out

    self.additional_outputs = []
    
    def _slice_last_input(x): return x[:, -1, :]
    tf_last_input = Lambda(_slice_last_input, name='last_user_input')(tf_input)
    tf_user_intent = Dense(units=128, activation='relu', name='user_intent_dense1')(tf_last_input)
    tf_user_intent = Dense(units=self.nr_user_labels, activation='softmax', name='user_intent_logits')(tf_user_intent)
    self.additional_outputs.append(tf_user_intent)

    self.has_bot_intent = False
    if 'BOT_INTENT' in configuration:
      self.has_bot_intent = bool(configuration['BOT_INTENT'])
      if self.has_bot_intent:
        inp_feats_bot_intent_decoder = tf_out
        if type(tf_out) == list: inp_feats_bot_intent_decoder = tf_out[0]
        tf_bot_intent_out = Dense(units=128, activation='relu', name='bot_intent_dense1')(inp_feats_bot_intent_decoder)
        tf_bot_intent_out = Dense(units=self.nr_bot_labels, activation='softmax', name='bot_intent_logits')(tf_bot_intent_out)
        self.additional_outputs.append(tf_bot_intent_out)
    #endif

    if self.is_timedistributed:
      self.trainable_model = Model(inputs=additional_inputs + list(crt_inputs.values()),
                                   outputs=tf_out + self.additional_outputs)

      self._log("Parent/encoder model\n{}"
                .format(self.logger.GetKerasModelSummary(self.trainable_model)))

      self.enc_pred_model = self.trainable_model
      
      self.num_transmisible_outputs = len(self.additional_outputs)

    return tf_out

  def CreateDecoder(self, configuration):
    if configuration == {}: return

    self.has_decoder = True

    assert self._use_keras
    tf_input, crt_inputs = self.CreateInputsAndEmbeddings(configuration)
    self.dec_tf_rec_input = tf_input
    self.dec_tf_inputs = list(crt_inputs.values())
    tf_out = self.CreateRecurrentCell(configuration, tf_input, save_lstm_cells=True)

    if 'READOUT' in configuration:
      self.decoder_readout = self.DenseNet(configuration['READOUT'], prefix='readout')
      tf_out = self.decoder_readout(tf_out)

    if self.is_timedistributed:
      additional_inputs = self.trainable_model.inputs

      decoder = Model(inputs=additional_inputs + list(crt_inputs.values()), outputs=tf_out)    
      self._log("Decoder model\n{}".format(self.logger.GetKerasModelSummary(decoder)))

      self.trainable_model = Model(inputs=list(self.input_tensors.values()),
                                   outputs=[tf_out] + self.enc_pred_model.outputs[-self.num_transmisible_outputs:])    
      self._log("End-to-end model\n{}".format(self.logger.GetKerasModelSummary(self.trainable_model)))

    return tf_out


  def DenseNet(self, conf, prefix='readout'):
    assert prefix in ['readout', 'bottleneck']
    units = conf['UNITS']
    act = conf['ACTIVATION']
    if act == 'None': act = None
    name = prefix + '_' + str(act)
    return Dense(units=units, name=name, activation=act)


  def CreateInputsAndEmbeddings(self, configuration, func_child=None):
    config_inputs = configuration['INPUTS']
    try:
      config_embeddings = configuration['EMBEDDINGS']
    except:
      config_embeddings = []
    identifier = configuration['IDENTIFIER']

    crt_inputs = {}
    all_feats = []
    extra_feats = []
    for inp in config_inputs:
      name = inp['NAME']
      has_batch_shape = 'BATCH_SHAPE' in inp

      if has_batch_shape:
        shape = self._convert_shape(inp['BATCH_SHAPE'])
      else:
        shape = self._convert_shape(inp['SHAPE'])
      is_input_for_emb = self._is_input_for_embedding(name, config_embeddings)
      dtype = tf.int32 if is_input_for_emb else tf.float32
      is_feedable = bool(inp['IS_FEEDABLE'])
      is_timedistributed = False
      if 'TIMEDISTRIBUTED' in inp:
        is_timedistributed = bool(inp['TIMEDISTRIBUTED'])
      if is_timedistributed:
        self.is_timedistributed = True
        dtype = tf.int32
        assert func_child is not None

      with tf.name_scope(identifier):
        if has_batch_shape:
          tf_X = Input(name=name, batch_shape=shape, dtype=dtype)
        else:
          tf_X = Input(name=name, shape=shape, dtype=dtype)
      crt_inputs[name] = tf_X
      if is_feedable:
        k = len(self.input_tensors) + 1
        self.input_tensors[k] = tf_X
      if is_timedistributed:
        with tf.name_scope(identifier):
          tf_td = TimeDistributed(func_child)(tf_X)
          extra_feats.append(tf_td)
#          for idx_crt, out in enumerate(func_child.output):
#            if idx_crt == 0:
#              tf_td = TimeDistributed(Model(func_child.input, out))(tf_X)
#              extra_feats.append(tf_td)
#            elif idx_crt == 1:
#              self.peek_tensor = TimeDistributed(Model(func_child.input, out))(tf_X)
      elif not is_input_for_emb:
        all_feats.append(tf_X)
    #endfor

    for i,emb in enumerate(config_embeddings):
      input_dim, output_dim, weights, target_shape, trainable = None, None, None, None, True
      output_shape, name, reuse = None, None, None
      connected_to = emb['CONNECTED_TO']

      if 'NAME' in emb: name = emb['NAME']
      if 'OUTPUT_SHAPE' in emb: output_shape = emb['OUTPUT_SHAPE']
      if 'REUSE' in emb: reuse = emb['REUSE']
      if 'INPUT_DIM' in emb: input_dim = emb['INPUT_DIM']
      if 'OUTPUT_DIM' in emb: output_dim = emb['OUTPUT_DIM']
      if 'TRAINABLE' in emb: trainable = emb['TRAINABLE']
      if 'EMB_MATRIX_PATH' in emb:
        if emb['EMB_MATRIX_PATH'] != '':
          path = emb['EMB_MATRIX_PATH']
          if emb['USE_DRIVE']:
            path = self.logger.GetDataFile(path)

          weights = np.load(path)
          assert len(weights.shape) == 2
          if 'PAD' in emb:
            nr_paddings = emb['PAD']
            if nr_paddings:
              n_cols = weights.shape[1]
              weights = np.concatenate((weights, np.zeros((1, n_cols))))
              weights = np.concatenate((weights, np.random.uniform(low=-0.05, high=0.05,
                                                                   size=(nr_paddings-1, n_cols))))
              input_dim += nr_paddings
  
      if 'RESHAPE' in emb:
        target_shape = self._convert_shape(emb['RESHAPE'])
        if target_shape == (): target_shape = None
  
      
      with tf.name_scope(identifier):
        if (reuse is None) or (reuse == ''):
          use_keras_emb = False
          if 'USE_KERAS' in emb:
            if bool(emb['USE_KERAS']):
              use_keras_emb = True
              if weights is None:
                Emb = Embedding(input_dim=input_dim, output_dim=output_dim, trainable=trainable,
                                name=name)
              else:
                Emb = Embedding(input_dim=input_dim, output_dim=output_dim, weights=[weights],
                                trainable=trainable, name=name)
          #endif use_keras_emb

          if not use_keras_emb:
            Emb = self.EmbeddingLayer(output_shape=output_shape, input_dim=input_dim,
                                      output_dim=output_dim, weights=weights, name=name,
                                      trainable=trainable)
          #endif NOT use_keras_emb
          
          tf_emb = Emb(crt_inputs[connected_to])
          self.EmbLayers[name] = Emb
        else:
          tf_emb = self.EmbLayers[reuse](crt_inputs[connected_to])

        if target_shape is not None:
          tf_emb = Reshape(target_shape=target_shape, name='reshape_emb_'+str(i))

      extra_feats.append(tf_emb)
    #endfor
    
    if 'BOTTLENECK_EMBEDDINGS' in configuration:
      with tf.name_scope(identifier):
        Bottleneck = self.DenseNet(configuration['BOTTLENECK_EMBEDDINGS'],
                                 prefix='bottleneck')
        tf_emb_feats = concatenate(extra_feats, name='input_bottleneck_embs')
        tf_emb_feats = Bottleneck(tf_emb_feats)
        all_feats.append(tf_emb_feats)
    else:
      all_feats += extra_feats
  
    if len(all_feats) > 1:
      tf_feats = concatenate(all_feats, name=identifier+'_input_feats_and_embs')
    else:
      tf_feats = all_feats[0]

    return tf_feats, crt_inputs


  def CreateRecurrentCell(self, configuration, tf_input, save_state=False, save_lstm_cells=False):
    """
      Parameters:
        - configuration: specifies how many layers a recurrent cell has and
                         how each layer is configurated.
                         More precisely, configuration is dict with the following keys:
                           IDENTIFIER: the name/identifier of the entire recurrent cell
                           LAYERS: a list of dictionaries,
                             each dictionary having the following keys: NAME (str), NR_UNITS (int),
                             TYPE (BIDIRECTIONAL / UNIDIRECTIONAL), SKIP_CONNECTIONS
                             Comming soon: The type of the recurrent cell (RNN, GRU, LSTM),
                             dropout and residual connections; Most important: functionality to be
                             used with GridSearch funtion in logger
    """
    lstm_sequences = {}
    crt_tf_input = tf_input
    identifier = configuration['IDENTIFIER']

    layers = configuration['LAYERS']
    for i in range(len(layers)):
      layer_dict = configuration['LAYERS'][i]
      name = layer_dict['NAME']
      units = layer_dict['NR_UNITS']
      lstm_type = layer_dict['TYPE'].lower()
      str_initial_state = ''
      if 'INITIAL_STATE' in layer_dict:
        str_initial_state = layer_dict['INITIAL_STATE']

      ### Check lstm_type
      if lstm_type not in valid_lstms:
        str_err = "ERROR! [Layer '{}'] The specified type ('{}') is not valid.".format(name, lstm_type)
        self._log(str_err)
        raise Exception(str_err)
      #endif

      ### Check initial_state
      initial_state = None
      if str_initial_state != "":
        if str_initial_state not in self.enc_layers_full_state.keys():
          self._log("[Layer '{}'] The specified initial_state ('{}') does not exist."
                    .format(name, str_initial_state))
          initial_state = None
        else:
          initial_state = self.enc_layers_full_state[str_initial_state]
      #endif

      ### Check skip connections
      all_skip_connections = [crt_tf_input]
      for skip in layer_dict['SKIP_CONNECTIONS']:
        if skip.upper() == 'INPUT':
          all_skip_connections.append(tf_input)
        else:
          skip = identifier + '_' + skip
          if skip not in lstm_sequences.keys():
            str_err = "ERROR! [Layer '{}'] The specified skip connection ('{}') does not exist.".format(name, skip)
            self._log(str_err)
            raise Exception(str_err)
          #endif
          all_skip_connections.append(lstm_sequences[skip])
      #endfor

      if len(all_skip_connections) >= 2:
        crt_tf_input = concatenate(all_skip_connections, name='skip_concat')
      
      name += ('_' + str(units))

      if lstm_type == 'bidirectional':
        # TOOD initial_state
        LSTMCell = Bidirectional(CuDNNLSTM(units=units, return_sequences=True, return_state=True), name=name)
        crt_tf_input, tf_state_h_fw, tf_state_c_fw, tf_state_h_bw, tf_state_c_bw = LSTMCell(crt_tf_input)
        tf_state_h = concatenate([tf_state_h_fw, tf_state_h_bw], name=identifier+'_concat_state_h_{}'.format(i))
        tf_state_c = concatenate([tf_state_c_fw, tf_state_c_bw], name=identifier+'_concat_state_c_{}'.format(i))
      
      if lstm_type == 'unidirectional':
        LSTMCell = CuDNNLSTM(units=units, return_sequences=True, return_state=True, name=name)
        crt_tf_input, tf_state_h, tf_state_c = LSTMCell(crt_tf_input, initial_state=initial_state)
      
      if save_state:
        self.enc_layers_full_state[layer_dict['NAME']] = [tf_state_h, tf_state_c]
        self.enc_full_state += [tf_state_h, tf_state_c]
      
      if save_lstm_cells:
        self.dec_lstm_cells.append(LSTMCell)
      
      lstm_sequences[name] = crt_tf_input
    #endfor

    return_state = 0
    if 'RETURN_STATE' in layers[-1]:
      return_state = layers[-1]['RETURN_STATE']

    if return_state == 0:
      return crt_tf_input
    elif return_state == 1:
      return tf_state_h
    else:
      return [tf_state_h, tf_state_c]


  def DefineTrainableModel(self):
    child1_config = self.encoder_architecture['CHILD1']
    child2_config = self.encoder_architecture['CHILD2']
    parent_config = self.encoder_architecture['PARENT']

    self.metrics = OrderedDict()
    for k,v in self.logger.get_K_metrics().items():
      if k in self._metrics_config: self.metrics[k] = v

    str_msg = 'K_model' if self._use_keras else 'tf_graph'
    self._log("Initializing '{}' with the following parameters:".format(str_msg))
    self._log(" Loss:'{}'  Optimizer:'{}'  LR:{}"
              .format(self._str_loss, self._str_optimizer, self._learning_rate))

    if not self._use_keras:
      self.tf_graph = tf.Graph()
      with self.tf_graph.as_default():
        func_child = self.CreateChildNet(child1_config)
        self.tf_out = self.CreateParentNet(parent_config, func_child)
        
        decoder_out = self.CreateDecoder(self.decoder_architecture)
        if decoder_out is not None: self.tf_out = decoder_out

        self.tf_y = tf.placeholder(dtype=tf.float32, shape=self.tf_out.shape, name='labels')
        self.tf_metrics = []
        for k,v in self.metrics.items():
          self.tf_metrics.append(v(self.tf_y, self.tf_out, use_tf_keras=True))
        
        self.tf_loss = self.logger.get_tf_loss(self._str_loss)(self.tf_y, self.tf_out)
        optimizer = self.logger.get_tf_optimizer(self._str_optimizer)(learning_rate=self._learning_rate)
        self.train_step = optimizer.minimize(self.tf_loss)
        self.init = tf.global_variables_initializer()
    else:
      func_child = self.CreateChildNet([child1_config, child2_config])
      self.CreateParentNet(parent_config, func_child)
      self.CreateDecoder(self.decoder_architecture)
      
      losses = [self._str_loss]
      lossWeights = [1.0]
      
      for _ in range(self.num_transmisible_outputs):
        losses.append("sparse_categorical_crossentropy")
        lossWeights.append(1.0)

      self.trainable_model.compile(optimizer=self._str_optimizer,
                                   loss=losses,
                                   loss_weights=lossWeights,
                                   metrics=list(self.metrics.values()))

    self._log("{} initialized!".format(str_msg))
    return


  def _fit_tf_graph(self, generator, nr_epochs, steps_per_epoch, validation_generator=None,
                    validation_steps=None, save_period=None, check_last_validation_ts=True):
    self.session = tf.Session(graph=self.tf_graph)
    self.session.run(self.init, options=tf.RunOptions(report_tensor_allocations_upon_oom = True))
    
    fn = self.logger.file_prefix + '_' + self._model_name + '/tf_graph'
    path = os.path.join(self.logger.GetModelsFolder(), fn)
    with self.tf_graph.as_default():
      self.logger.SaveTFGraphCheckpoint(session=self.session,
                                    placeholders=list(self.input_tensors.values()) + [self.tf_y],
                                    operations=[self.train_step, self.tf_loss, self.tf_out] + self.tf_metrics,
                                    epoch=0,
                                    save_path=path)

    str_metrics = list(self.metrics.keys())

    for epoch in range(nr_epochs):
      self._log("Epoch {}/{}".format(epoch+1, nr_epochs))
      loss_hist = []
      histories = [[]] * len(str_metrics)

      t = trange(steps_per_epoch, desc='', leave=True)
      for step in t:
        batch = next(generator)
        X = batch[0]
        y = batch[1]

        assert len(self.input_tensors) == len(X)

        feed_dict = {}
        for i,(k,v) in enumerate(self.input_tensors.items()):
          feed_dict[v] = X[i]
        #endfor
        feed_dict[self.tf_y] = y

        result = self.session.run([self.train_step, self.tf_loss] + self.tf_metrics,
                                  feed_dict=feed_dict)
        loss = result[1]
        loss_hist.append(loss)
        descr = "Loss: {:.3f}".format(loss)

        if len(self.tf_metrics) > 0:
          val_metrics = result[2:] 
          for i,v in enumerate(val_metrics):
            histories[i].append(v)
            descr += "  {}: {:.3f}".format(str_metrics[i], v)

        t.set_description(descr)
        t.refresh()
      #endfor
      
      str_results = "Loss: {:.3f}".format(np.array(loss_hist).mean())
      for i,h in enumerate(histories):
        str_results += "  {}: {:.3f}".format(str_metrics[i], np.array(h).mean())
      self._log("Mean results - " + str_results)
      
      if validation_generator is None: continue
      
      assert validation_steps is not None
      
      if (epoch+1) % save_period == 0:
        ### Validation logic
        loss_hist = []
        histories = [[]] * len(str_metrics)
        real_hist = []
        pred_hist = []
        t = trange(validation_steps, desc='', leave=True)
        for step in t:
          batch = next(validation_generator)
          X = batch[0]
          y = batch[1]
          assert len(self.input_tensors) == len(X)
          feed_dict = {}
          for i,(k,v) in enumerate(self.input_tensors.items()):
            feed_dict[v] = X[i]
          #endfor
          feed_dict[self.tf_y] = y
  
          result = self.session.run([self.tf_out, self.tf_loss] + self.tf_metrics,
                                    feed_dict=feed_dict)
          out = result[0]
          pred_hist.append(out)
          real_hist.append(y)
          loss = result[1]
          loss_hist.append(loss)
          descr = "Val_loss: {:.3f}".format(loss)
  
          if len(self.tf_metrics) > 0:
            val_metrics = result[2:] 
            for i,v in enumerate(val_metrics):
              histories[i].append(v)
              descr += "  Val_{}: {:.3f}".format(str_metrics[i], v)
          #endif
          
          t.set_description(descr)
          t.refresh()
        #endfor
        str_results = "Val_loss: {:.3f}".format(np.array(loss_hist).mean())
        for i,h in enumerate(histories):
          str_results += "  Val_{}: {:.3f}".format(str_metrics[i], np.array(h).mean())
        self._log("Mean validation results - " + str_results)
        
        if check_last_validation_ts:
          real_hist = list(map(lambda x: np.squeeze(x)[-1], real_hist))
          pred_hist = list(map(lambda x: np.squeeze(x)[-1], pred_hist))
          from sklearn.metrics import mean_absolute_error
          self._log("Validation results based on last timestep: MAE={:.3f}"
                    .format(mean_absolute_error(real_hist, pred_hist)))
  
        ### End Validation logic
        
        fn = self.logger.file_prefix + '_' + self._model_name + '/tf_graph'
        path = os.path.join(self.logger.GetModelsFolder(), fn)
        with self.tf_graph.as_default():
          self.logger.SaveTFGraphCheckpoint(session=self.session,
                                            placeholders=list(self.input_tensors.values()) + [self.tf_y],
                                            operations=[self.train_step, self.tf_loss, self.tf_out] + self.tf_metrics,
                                            epoch=epoch+1,
                                            save_path=path)
      #endif
    #endfor
    return

  def _fit_k_model(self, generator, nr_epochs, steps_per_epoch, validation_generator=None,
                   validation_steps=None, monitor='loss', mode='min', save_period=None):
    callbacks = []
    callbacks.append(self.logger.GetKerasEpochCallback(monitor=monitor, mode=mode))

    if save_period is not None:
      callbacks.append(self.logger.GetKerasCheckpointCallback(model_name=self._model_name,
                                                              period=save_period,
                                                              monitor=monitor,
                                                              mode=mode))

    self.trainable_model.fit_generator(generator=generator, epochs=nr_epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       callbacks=callbacks,
                                       validation_data=validation_generator,
                                       validation_steps=validation_steps)
    return

  def Fit(self, generator, nr_epochs, steps_per_epoch, validation_generator=None,
          validation_steps=None, monitor='loss', mode='min', save_period=None):
    self._log("Training model ...")
    if not self._use_keras:
      self._fit_tf_graph(generator=generator,
                         nr_epochs=nr_epochs,
                         steps_per_epoch=steps_per_epoch,
                         save_period=save_period,
                         validation_generator=validation_generator,
                         validation_steps=validation_steps)
    else:
      if self.has_decoder:
        self._fit_encoder_decoder(generator=generator,
                                  nr_epochs=nr_epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  save_period=save_period,
                                  validation_generator=validation_generator,
                                  validation_steps=validation_steps,
                                  monitor=monitor,
                                  mode=mode)
      else:
        self._fit_k_model(generator=generator,
                          nr_epochs=nr_epochs,
                          steps_per_epoch=steps_per_epoch,
                          save_period=save_period,
                          validation_generator=validation_generator,
                          validation_steps=validation_steps,
                          monitor=monitor,
                          mode=mode)
      return
  
  
  def _fit_encoder_decoder(self, generator, nr_epochs, steps_per_epoch, validation_generator=None,
                           validation_steps=None, monitor='loss', mode='min', save_period=None):
    self.save_period = save_period
    self.loss_hist = []
    epoch_callback = self.logger.GetKerasEpochCallback(predef_callback=self._on_epoch_end_callback)
    reduce_lr_callback = ReduceLROnPlateau(monitor="loss", factor=0.5, 
                                           patience=2, min_lr=0.00001,
                                           verbose=1, mode='min')
    
    if self.data_processer.validate:
      self.s2s_metrics = ['BLEU_ARGMAX', 'BLEU_SAMPLING', 'ACC_INTENTS_USER', 'ACC_INTENT_BOT']
      self.dict_global_results_train = {}
      self.dict_global_results_val   = {}
      self.dict_global_results_train['EPOCH'] = []
      self.dict_global_results_val['EPOCH'] = []
      
      for i,m in enumerate(self.s2s_metrics):
        if i < len(self.s2s_metrics) - 1:
          self.dict_global_results_train[m] = []
          self.dict_global_results_val[m] = []
        elif self.has_bot_intent:
          self.dict_global_results_train[m] = []
          self.dict_global_results_val[m] = []
        
        
      
    self.trainable_model.fit_generator(generator=generator, epochs=nr_epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       callbacks=[epoch_callback, reduce_lr_callback],
                                       validation_data=validation_generator,
                                       validation_steps=validation_steps)
    
    if self.data_processer.validate:
      self.logger.SaveDataFrame(pd.DataFrame.from_dict(self.dict_global_results_train),
                                fn='global_train_results',
                                show_prefix=True,
                                to_data=False,
                                ignore_index=True)
      self.logger.SaveDataFrame(pd.DataFrame.from_dict(self.dict_global_results_val),
                                fn='global_val_results',
                                show_prefix=True,
                                to_data=False,
                                ignore_index=True)

    return

  def _on_epoch_end_callback(self, epoch, logs):
    epoch = epoch + 1 + self.epoch_loaded_model
    str_logs = ""
    for key,val in logs.items():
      str_logs += "{}:{:.6f}  ".format(key,val)
    self._log(" Train/Fit: Epoch: {} Results: {}".format(epoch,str_logs))

    validation_epochs = None
    if 'VALIDATION_EPOCHS' in self.config_data:
      validation_epochs = self.config_data['VALIDATION_EPOCHS']
    
    if validation_epochs is not None and self.data_processer.validate:
      if (epoch % validation_epochs == 0) or epoch == 1:
        self.dict_global_results_train['EPOCH'].append(epoch)
        self.dict_global_results_val['EPOCH'].append(epoch)
        
        self._log("Validating ...")
        self.Predict(dataset='train')
        self._log("'train' global explanatory results:\n{}".format(pd.DataFrame.from_dict(self.dict_global_results_train).to_string()))
        self.Predict(dataset='validation')
        self._log("'validation' global explanatory results:\n{}".format(pd.DataFrame.from_dict(self.dict_global_results_val).to_string()))
    
    loss = logs['loss']
    self.loss_hist.append((epoch, loss))    
    self._save_model(epoch, loss)
    return


  def _mk_plot(self, fig_path):
    sns.set()
    plt.figure(figsize=(1280/96, 720/96), dpi=96)
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    epochs = [i[0] for i in self.loss_hist]
    losses = [i[1] for i in self.loss_hist]
    
    plt.plot(epochs, losses, linestyle='--', marker='o', color='r')
    plt.savefig(fig_path, dpi=96)
    plt.close()
    return

  def _save_model(self, epoch, loss):
    if self.save_period is None:
      return
    if epoch != 1:  ## epoch 1 is always saved in order to see progress
      if (epoch % self.save_period) != 0:
        return
      
    assert len(self.model_trained_layers) > 0, 'Unknown list of trained layers!'

    str_epoch = str(epoch).zfill(2)
    str_loss = "{:.2f}".format(loss)
    model_folder = self._model_name + "_epoch" + str_epoch + "_loss" + str_loss

    model_full_path = os.path.join(self.logger.GetModelsFolder(), model_folder)
    if not os.path.exists(model_full_path):
      os.makedirs(model_full_path)

    fn_weights = model_folder + '/weights'
    fn_config = os.path.join(self.logger.GetModelsFolder(), model_folder + '/config.txt')
    fn_losshist = os.path.join(self.logger.GetModelsFolder(), model_folder + '/loss_hist.jpg')
    self.config_data['EPOCH'] = epoch
    with open(fn_config, 'w') as f:
      f.write(json.dumps(self.config_data, indent=2))
    self._log("Saved model config in '{}'.".format(fn_config))
    
    true_names = []
    for lyr in self.trainable_model.layers:
      if len(lyr.trainable_weights) > 0:
        true_names.append(lyr.name)
    
    diff = list(set(true_names) - set(self.model_trained_layers))
    if len(diff) != 0: self._log("WARNING! model_trained_layers mismatch. Check {}".format(diff))

    self.logger.SaveKerasModelWeights(fn_weights, self.trainable_model, self.model_trained_layers)
    self._mk_plot(fn_losshist)
    return
  

  def LoadModelWeightsAndConfig(self, model_label):
    fn_config = os.path.join(self.logger.GetModelsFolder(), model_label + '/config.txt')
    if not os.path.exists(fn_config):
      self._log("Did not found the specified model_label: '{}'.".format(model_label))
      return
    
    self.logger._configure_data_and_dirs(fn_config, config_file_encoding=None)
    self.config_data = self.logger.config_data
    self._parse_config_data()
    self.epoch_loaded_model = self.config_data['EPOCH']

    assert len(self.model_trained_layers) > 0, 'Unknown list of trained layers!'
    self.DefineTrainableModel()

    fn_weights = model_label + '/weights'
    self.logger.LoadKerasModelWeights(fn_weights, self.trainable_model, self.model_trained_layers)
    
    self.CreatePredictionModels()
    return



  def CreatePredictionModels(self):
    self._log("Creating prediction models ...")
    self.enc_pred_model = Model(self.enc_tf_inputs, self.enc_full_state + self.additional_outputs)
    
    crt_tf_input = self.dec_tf_rec_input
    tf_input = self.dec_tf_rec_input
    dec_model_inputs = []
    dec_model_outputs = []
    lstm_sequences = {}
    for i in range(len(self.dec_lstm_cells)):
      layer_dict = self.decoder_architecture['LAYERS'][i]
      name = layer_dict['NAME']
      units = layer_dict['NR_UNITS']
      
      tf_inp_h = Input((units,), name='gen_inp_h_' + str(i+1))
      tf_inp_c = Input((units,), name='gen_inp_c_' + str(i+1))
      dec_model_inputs.append(tf_inp_h)
      dec_model_inputs.append(tf_inp_c)
      
      ### Check skip connections
      all_skip_connections = [crt_tf_input]
      for skip in layer_dict['SKIP_CONNECTIONS']:
        if skip.upper() == 'INPUT':
          all_skip_connections.append(tf_input)
        else:
          if skip not in lstm_sequences.keys():
            str_err = "ERROR! [DecLayer '{}'] The specified skip connection ('{}') does not exist.".format(name, skip)
            self._log(str_err)
            raise Exception(str_err)
          #endif
          all_skip_connections.append(lstm_sequences[skip])
      #endfor
      
      if len(all_skip_connections) >= 2: crt_tf_input = concatenate(all_skip_connections)
      
      crt_tf_input, tf_h, tf_c = self.dec_lstm_cells[i](crt_tf_input,
                                                        initial_state=[tf_inp_h, tf_inp_c])
      
      dec_model_outputs.append(tf_h)
      dec_model_outputs.append(tf_c)
      
      lstm_sequences[name] = crt_tf_input
    #endfor
    
    tf_gen_preds = self.decoder_readout(crt_tf_input)
    
    self.dec_pred_model = Model(inputs=dec_model_inputs + self.dec_tf_inputs,
                                outputs=dec_model_outputs + [tf_gen_preds])
    
    self._log("Enc pred model\n{}".format(self.logger.GetKerasModelSummary(self.enc_pred_model)))
    self._log("Dec pred model\n{}".format(self.logger.GetKerasModelSummary(self.dec_pred_model)))
    return
  
  

  def _step_by_step_prediction(self, _input, method='sampling', verbose=1, return_text=True):
    assert method in ['sampling', 'argmax', 'beamsearch']
    _type = type(_input)

    if _type in [str, list]:
      if type(_input) is list: _input = '\n'.join(_input)
      input_tokens = self.data_processer.input_word_text_to_tokens(_input, use_characters=True)
      input_tokens = np.expand_dims(np.array(input_tokens), axis=0)
      str_input = _input
    elif _type is np.ndarray:
      input_tokens = _input
      input_tokens = np.expand_dims(input_tokens, axis=0)
      str_input = self.data_processer.translate_tokenized_input(_input)

    if verbose: self._log("Given '{}' the decoder predicted:".format(str_input))
    predict_results = self.enc_pred_model.predict(input_tokens)
    enc_states = predict_results[:-1]

    idx_last_intent_user = -2 if self.has_bot_intent else -1

    last_intent_user = np.expand_dims(np.argmax(predict_results[idx_last_intent_user], axis=-1), axis=-1)    
    intent_bot = None
    if self.has_bot_intent:
      intent_bot = np.expand_dims(np.argmax(predict_results[-1], axis=-1), axis=-1)
    
    dec_model_inputs = []

    for i in range(len(self.decoder_architecture['LAYERS'])):
      layer_dict = self.decoder_architecture['LAYERS'][i]
      units = layer_dict['NR_UNITS']
      try:
        enc_layer_initial_state = layer_dict['INITIAL_STATE']
      except:
        enc_layer_initial_state = ""
      inp_h = np.zeros((1, units))
      inp_c = np.zeros((1, units))

      if enc_layer_initial_state != "":
        idx = self._get_key_index_odict(self.enc_layers_full_state, enc_layer_initial_state)
        if idx is not None:
          inp_h, inp_c = enc_states[2*idx:2*(idx+1)]
          if verbose: self._log("Enc_h: {}  Dec_h: {}".format(inp_h.sum(), inp_c.sum()))
      #endif

      dec_model_inputs += [inp_h, inp_c]
    #endfor

    current_gen_token = self.data_processer.start_char_id
    predicted_tokens = []
    nr_preds = 0
    while current_gen_token != self.data_processer.end_char_id:
      current_gen_token = np.array(current_gen_token).reshape((1,1))

      if not self.has_bot_intent:
        dec_model_outputs = self.dec_pred_model.predict(dec_model_inputs + [current_gen_token, last_intent_user])
      else:
        dec_model_outputs = self.dec_pred_model.predict(dec_model_inputs + [current_gen_token, last_intent_user, intent_bot])

      P = dec_model_outputs[-1].squeeze()
      if method == 'sampling':
        current_gen_token = np.random.choice(range(P.shape[0]), p=P)
      if method == 'argmax':
        current_gen_token = np.argmax(P)
      if method == 'beamsearch':
        raise Exception("{} not implemented yet.".format(method))  ### TODO implement beamsearch

      predicted_tokens.append(current_gen_token)
      dec_model_inputs = dec_model_outputs[:-1]
      nr_preds += 1
      if nr_preds == 50:
        break
    #end_while
    predicted_tokens = predicted_tokens[:-1]
    
    if return_text:
      predicted_text = self.data_processer.input_word_tokens_to_text([predicted_tokens])
      if verbose:
        self._log("  --> '{}'".format(predicted_text))
    else:
      predicted_text = list(map(lambda x: self.data_processer.dict_id2word[x], predicted_tokens))

    if self.has_bot_intent: intent_bot = intent_bot.reshape(-1)
    last_intent_user = last_intent_user.reshape(-1)

    return predicted_text, last_intent_user, intent_bot

  
  def compute_metrics(self, bleu_params, intents_user_params=None, intent_bot_params=None):
    references, candidates = bleu_params
    bleu = compute_bleu(references, candidates)
    
    acc_intents_user, acc_intent_bot = None, None
    
    if intents_user_params is not None:
      true_intents_user, predicted_intents_user = intents_user_params
      acc_intents_user = accuracy_score(true_intents_user, predicted_intents_user)

    if intent_bot_params is not None:
      true_intent_bot, predicted_intent_bot = intent_bot_params
      acc_intent_bot = accuracy_score(true_intent_bot, predicted_intent_bot)

    return bleu, acc_intents_user, acc_intent_bot
    


  def Predict(self, dataset):
    dataset = dataset.lower()
    assert dataset in ['train', 'validation']
    dict_results = {}
    
    if dataset == 'train':
      ds = self.data_processer.batches_train_to_validate
    elif dataset == 'validation':
      ds = self.data_processer.batches_validation
    #endif
    
    for nr_turns in ds.keys():
      dict_results[nr_turns] = {}
      for i,m in enumerate(self.s2s_metrics): 
        if i < len(self.s2s_metrics) - 1: dict_results[nr_turns][m] = []
        elif self.has_bot_intent: dict_results[nr_turns][m] = []

    for nr_turns, all_dialogues in ds.items():
      for current_state in all_dialogues:
        if dataset == 'train':
          reference = [list(map(lambda x: self.data_processer.dict_id2word[x], current_state[1][1:-1]))]
          true_intent_user  = [current_state[2][-2]]
          true_intent_bot   = current_state[2][-1:]
        elif dataset == 'validation':
          reference = current_state[1]['STATEMENTS']
          true_intent_user  = current_state[2][-1:]
          true_intent_bot   = [current_state[1]['LABEL']]
        #endif

        prediction_result = self._step_by_step_prediction(current_state[0],
                                                          method='argmax', verbose=0,
                                                          return_text=False)
        candidate_argmax, predicted_intent_user, predicted_intent_bot = prediction_result
        candidate_argmax = [candidate_argmax] * len(reference)

        prediction_result = self._step_by_step_prediction(current_state[0],
                                                          method='sampling', verbose=0,
                                                          return_text=False)
        candidate_sampling, _, _ = prediction_result
        candidate_sampling = [candidate_sampling] * len(reference)

        intent_bot_params = None
        if predicted_intent_bot is not None:
          intent_bot_params = (true_intent_bot, predicted_intent_bot)

        bleu_argmax, acc_intents_user, acc_intent_bot = self.compute_metrics(
            bleu_params=(reference, candidate_argmax),
            intents_user_params=(true_intent_user, predicted_intent_user),
            intent_bot_params=intent_bot_params) 

        bleu_sampling, _, _ = self.compute_metrics((reference, candidate_sampling))

        dict_results[nr_turns]['BLEU_ARGMAX'].append(bleu_argmax)
        dict_results[nr_turns]['BLEU_SAMPLING'].append(bleu_sampling)
        dict_results[nr_turns]['ACC_INTENTS_USER'].append(acc_intents_user)
        if self.has_bot_intent: dict_results[nr_turns]['ACC_INTENT_BOT'].append(acc_intent_bot)
        
        rand = random.randint(0,7)
        if dataset == 'validation' and rand == 0:
          str_dialogue = '[' + "\n".join(current_state[0]) + ']'
          str_candidate_argmax = '[' + " ".join(candidate_argmax[0]) + ']'
          str_candidate_sampling = '[' + " ".join(candidate_sampling[0]) + ']'
          self._log('Given\n{}'.format(str_dialogue))
          self._log("the decoder predicted (argmax):\n{}".format(str_candidate_argmax))
          self._log("the decoder predicted (sampling):\n{}".format(str_candidate_sampling))
      #endfor

      for i,m in enumerate(self.s2s_metrics): 
        if i < len(self.s2s_metrics) - 1: dict_results[nr_turns][m] = np.mean(dict_results[nr_turns][m])
        elif self.has_bot_intent: dict_results[nr_turns][m] = np.mean(dict_results[nr_turns][m])
      #endfor
    #endfor
    
    df_results = pd.DataFrame.from_dict(dict_results)

    self._log("'{}' granular explanatory results:\n{}".format(dataset, df_results.to_string()))

    mean_results = df_results.mean(axis=1)
    if dataset == 'train':
      for i,m in enumerate(self.s2s_metrics): 
        if i < len(self.s2s_metrics) - 1: self.dict_global_results_train[m].append(mean_results[m])
        elif self.has_bot_intent: self.dict_global_results_train[m].append(mean_results[m])
      
    elif dataset == 'validation':
      for i,m in enumerate(self.s2s_metrics): 
        if i < len(self.s2s_metrics) - 1: self.dict_global_results_val[m].append(mean_results[m])
        elif self.has_bot_intent: self.dict_global_results_val[m].append(mean_results[m])

    return
  
  
  def Reset(self):
    K.clear_session()
    self._initialize_datastructures()