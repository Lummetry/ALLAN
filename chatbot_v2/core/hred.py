import tensorflow as tf
import numpy as np
from collections import OrderedDict

from libraries.generic_obj import LummetryObject


MAX_WORDS = 50
MAX_CHARACTERS = 100
NR_USER_LABELS = 60

REC_STACK_IDENTIFIER = 'identifier'
REC_STACK_LAYERS = 'layers'
REC_STACK_LYR_NAME = 'name'
REC_STACK_LYR_NR_UNITS = 'nr_units'
REC_STACK_LYR_TYPE = 'type'
REC_STACK_LYR_TYPE_BIDI = 'bidirectional'
REC_STACK_LYR_TYPE_UNI = 'unidirectional'
REC_STACK_LYR_INITIAL_STATE = 'initial_state'
REC_STACK_LYR_SKIP_CONNECTIONS = 'skip_connections'
REC_STACK_LYR_SKIP_CONNECTIONS_INPUT = 'skip_input'
REC_STACK_LYR_RETURN_STATE = 'return_state'


class HRED(LummetryObject):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    
    self.enc_layers_full_state = OrderedDict()
    self.enc_full_state = []
    self.dec_lstm_cells = []
    
    self.embedding = None
    
    self.sent_level_model = None
    self.conv_level_model = None
    self.enc_pred_model = None
    
    return
  
  def _create_recurrent_stack(self, configuration, tf_input, save_state=False,
                              save_lstm_cells=False):
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
    identifier = configuration[REC_STACK_IDENTIFIER]

    layers = configuration[REC_STACK_LAYERS]
    for i in range(len(layers)):
      layer_dict = layers[i]
      name = layer_dict[REC_STACK_LYR_NAME]
      units = layer_dict[REC_STACK_LYR_NR_UNITS]
      lstm_type = layer_dict[REC_STACK_LYR_TYPE]
      
      str_initial_state = ""
      if REC_STACK_LYR_INITIAL_STATE in layer_dict:
        str_initial_state = layer_dict[REC_STACK_LYR_INITIAL_STATE]

      ### Check initial_state
      initial_state = None
      if str_initial_state != "":
        if str_initial_state not in self.enc_layers_full_state.keys():
          self.P("[Layer '{}'] The specified initial_state ('{}') does not exist."
                 .format(name, str_initial_state))
          initial_state = None
        else:
          initial_state = self.enc_layers_full_state[str_initial_state]
      #endif

      ### Check skip connections
      all_skip_connections = [crt_tf_input]
      for skip in layer_dict[REC_STACK_LYR_SKIP_CONNECTIONS]:
        if skip == REC_STACK_LYR_SKIP_CONNECTIONS_INPUT:
          all_skip_connections.append(tf_input)
        else:
          if skip not in lstm_sequences.keys():
            str_err = "ERROR! [Layer '{}'] The specified skip connection ('{}') does not exist.".format(name, skip)
            self.P(str_err)
            raise ValueError(str_err)
          #endif
          all_skip_connections.append(lstm_sequences[skip])
      #endfor

      if len(all_skip_connections) >= 2:
        crt_tf_input = tf.keras.layers.concatenate(all_skip_connections,
                                                   name='skip_concat')
      

      if lstm_type == REC_STACK_LYR_TYPE_BIDI:
        # TOOD initial_state
        if save_state:
          _layer = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True)
          LSTMCell = tf.keras.layers.Bidirectional(_layer, name=name + '_' + str(units))
          crt_tf_input, tf_state_h_fw, tf_state_c_fw, tf_state_h_bw, tf_state_c_bw = LSTMCell(crt_tf_input)
          tf_state_h = tf.keras.layers.concatenate([tf_state_h_fw, tf_state_h_bw],
                                                   name=identifier+'_concat_state_h_{}'.format(i))
          tf_state_c = tf.keras.layers.concatenate([tf_state_c_fw, tf_state_c_bw],
                                                   name=identifier+'_concat_state_c_{}'.format(i))
        else:
          if i == len(layers) - 1:
            _layer = tf.keras.layers.LSTM(units=units, return_sequences=False)
            LSTMCell = tf.keras.layers.Bidirectional(_layer, name=name + '_' + str(units))
            tf_state_h = LSTMCell(crt_tf_input)
          else:
            _layer = tf.keras.layers.LSTM(units=units, return_sequences=True)
            LSTMCell = tf.keras.layers.Bidirectional(_layer, name=name + '_' + str(units))
            crt_tf_input = LSTMCell(crt_tf_input)
      
      if lstm_type == REC_STACK_LYR_TYPE_UNI:
        LSTMCell = tf.keras.layers.LSTM(units=units,
                                        return_sequences=True,
                                        return_state=True,
                                        name=name + '_' + str(units))
        crt_tf_input, tf_state_h, tf_state_c = LSTMCell(crt_tf_input, initial_state=initial_state)
      
      if save_state:
        self.enc_layers_full_state[name] = [tf_state_h, tf_state_c]
        self.enc_full_state += [tf_state_h, tf_state_c]
      
      if save_lstm_cells:
        self.dec_lstm_cells.append(LSTMCell)
      
      lstm_sequences[name] = crt_tf_input
    #endfor

    return_state = 0
    if REC_STACK_LYR_RETURN_STATE in layers[-1]:
      return_state = layers[-1][REC_STACK_LYR_RETURN_STATE]

    if return_state == 0:
      return crt_tf_input
    elif return_state == 1:
      return tf_state_h
    else:
      return [tf_state_h, tf_state_c]


  def _create_recurrent_stack_config(self, identifier: str, units: str,
                                     return_state: int, skip_connections: bool,
                                     bidirectional: bool):
    units = list(map(lambda x: int(x), units.split(',')))
    configuration = {}
    configuration[REC_STACK_IDENTIFIER] = identifier
    configuration[REC_STACK_LAYERS] = []
    
    stack_names = [REC_STACK_LYR_SKIP_CONNECTIONS_INPUT]
    for i in range(len(units)):
      name = identifier + '_lstm_{}'.format(i+1)
      dct_lyr = {}
      dct_lyr[REC_STACK_LYR_NAME] = name
      dct_lyr[REC_STACK_LYR_NR_UNITS] = units[i]
      dct_lyr[REC_STACK_LYR_TYPE] = REC_STACK_LYR_TYPE_UNI
      if bidirectional:
        dct_lyr[REC_STACK_LYR_TYPE] = REC_STACK_LYR_TYPE_BIDI
    
      dct_lyr[REC_STACK_LYR_RETURN_STATE] = 0
      if i == len(units) - 1:
        dct_lyr[REC_STACK_LYR_RETURN_STATE] = return_state
      
      dct_lyr[REC_STACK_LYR_SKIP_CONNECTIONS] = []
      if skip_connections and len(stack_names) >= 2:
        dct_lyr[REC_STACK_LYR_SKIP_CONNECTIONS] = [stack_names[-2]]
      
      stack_names.append(name)
      
      configuration[REC_STACK_LAYERS].append(dct_lyr)
    #endfor

    return configuration
    
  def create_sentence_level_model(self):
    
    tf_input = tf.keras.layers.Input(shape=(MAX_WORDS + MAX_CHARACTERS, ),
                                     name='inp_sent_words_chars')

    def _slice(x, start, end):
      return x[:, start:end]

    
    start, end = 0, MAX_WORDS
    tf_input_words = tf.keras.layers.Lambda(_slice, arguments={'start': start, 'end': end})(tf_input)
    
    start, end = MAX_WORDS, MAX_WORDS + MAX_CHARACTERS
    tf_input_chars = tf.keras.layers.Lambda(_slice, arguments={'start': start, 'end': end})(tf_input)

    inp_dim_emb_words = 100000
    out_dim_emb_words = 128
    inp_dim_emb_chars = 128
    out_dim_emb_chars = 16

    self.embedding = {'words': tf.keras.layers.Embedding(input_dim=inp_dim_emb_words,
                                                         output_dim=out_dim_emb_words,
                                                         trainable=True,
                                                         name='emb_words'),
                      
                      'chars': tf.keras.layers.Embedding(input_dim=inp_dim_emb_chars,
                                                         output_dim=out_dim_emb_chars,
                                                         trainable=True,
                                                         name='emb_chars')}
    
    

    tf_emb_words = self.embedding['words'](tf_input_words)
    tf_emb_chars = self.embedding['chars'](tf_input_chars)

    config_recurrent_stack_words_enc = self._create_recurrent_stack_config(identifier='child_words',
                                                                           units='128',
                                                                           return_state=1,
                                                                           skip_connections=True,
                                                                           bidirectional=True)

    config_recurrent_stack_chars_enc = self._create_recurrent_stack_config(identifier='child_chars',
                                                                           units='128',
                                                                           return_state=1,
                                                                           skip_connections=True,
                                                                           bidirectional=True)

    tf_out_words = self._create_recurrent_stack(configuration=config_recurrent_stack_words_enc,
                                                tf_input=tf_emb_words)

    tf_out_chars = self._create_recurrent_stack(configuration=config_recurrent_stack_chars_enc,
                                                tf_input=tf_emb_chars)

    tf_out = tf.keras.layers.concatenate([tf_out_words, tf_out_chars],
                                         name='word_char_level')

    self.sent_level_model = tf.keras.Model(inputs=[tf_input],
                                outputs=[tf_out])
    self.P("Child model\n{}".format(self.log.get_keras_model_summary(self.sent_level_model)))
    return self.sent_level_model


  def create_conversation_level_model(self):
    tf_input = tf.keras.layers.Input(shape=(None, MAX_WORDS + MAX_CHARACTERS),
                                     dtype=np.int32, name='inp_words_chars')
    
    tf_all_sentences_encoded = tf.keras.layers.TimeDistributed(self.sent_level_model)(tf_input)
    
    config_recurrent_stack_sentences_enc = self._create_recurrent_stack_config(identifier='conv_level',
                                                                               units='128',
                                                                               return_state=2,
                                                                               skip_connections=True,
                                                                               bidirectional=True)
    
    tf_out = self._create_recurrent_stack(configuration=config_recurrent_stack_sentences_enc,
                                          tf_input=tf_all_sentences_encoded,
                                          save_state=True)
    
    def _slice_last_input(x): return x[:, -1, :]
    tf_last_user_sentence = tf.keras.layers.Lambda(_slice_last_input,
                                                   name='last_user_sentence')(tf_all_sentences_encoded)
    
    tf_user_intent = tf.keras.layers.Dense(units=128,
                                           activation='relu',
                                           name='user_intent_dense1')(tf_last_user_sentence)
    tf_user_intent = tf.keras.layers.Dense(units=NR_USER_LABELS,
                                           activation='softmax',
                                           name='user_intent_logits')(tf_user_intent)
    
    self.conv_level_model = tf.keras.Model(inputs=tf_input,
                                           outputs=[tf_out, tf_user_intent])

    self.P("Conversation level /encoder model\n{}"
           .format(self.log.get_keras_model_summary(self.conv_level_model)))

    self.enc_pred_model = self.conv_level_model
    
    return tf_out
    
  def create_decoder(self):
    pass    

  
if __name__ == '__main__':
  
  import argparse
  from libraries import Logger
  
  parser = argparse.ArgumentParser()
  parser.add_argument("-b", "--base_folder", help="Base folder for storage",
                      type=str, default='dropbox')
  parser.add_argument("-a", "--app_folder", help="App folder for storage",
                      type=str, default='_allan_data/_chatbot')
  
  args = parser.parse_args()
  base_folder = args.base_folder
  app_folder = args.app_folder
  
  log = Logger(lib_name='HRED', base_folder=base_folder, app_folder=app_folder)

  hred = HRED(log=log)
  func_child = hred.create_sentence_level_model()
  
  
  




  
    