# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:20:23 2019

@author: damia
"""

from tagger.brain.base_engine import ALLANTaggerEngine 
import tensorflow as tf

from libraries.lummetry_layers.gated import GatedDense

_VER_ = '0.8.1'

class ALLANTaggerCreator(ALLANTaggerEngine):
  """
  
  """
  def __init__(self,    
               inputs=None,
               outputs=None,
               columns_end=None,
               **kwargs):
    """
    pass either dicts or output_size/vocab_size
    embed_size also optional - will be loaded based on saved embeds
    """
    super().__init__(**kwargs)
    self.trained = False
    self.__version__ = _VER_
    self.__name__ = 'AT_MC'
    self.pre_inputs = inputs
    self.pre_outputs = outputs
    self.pre_columns_end = columns_end
    self.model_prepared = False
    return
  

    
  
  def _define_column(self, tf_input, kernel, filters, name, 
                     activation='relu', last_step='lstm', bn=True,
                     use_cuda=True, step=1,
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
    last_step = last_step.lower()
    if depth == 0:
      # TODO: smart infer depth of the column!
      depth = max(1, 8 // kernel)
    tf_x = tf_input
    if kernel != 0: # this is convs column
      for L in range(1, depth+1):
        tf_x = tf.keras.layers.Conv1D(filters=filters,
                                      kernel_size=kernel,
                                      strides=step,
                                      name=name+'_conv{}_{}'.format(kernel,L))(tf_x)
        if bn:
          tf_x = tf.keras.layers.BatchNormalization(name=name+'_bn{}'.format(L))(tf_x)
        tf_x = tf.keras.layers.Activation(activation, 
                                          name=name+'_{}{}'.format(activation,L))(tf_x)
    else:
      # this is a a  LSTM column
      if last_step != 'lstm':
        raise ValueError("For non-conv columns last step must be 'lstm'")
        
    if last_step == 'lstm':
      if use_cuda:
        nc = '_CUlsmt'
        lyr_last1 = tf.keras.layers.CuDNNLSTM(filters, name=name+nc)
      else:
        nc = '_lsmt'
        lyr_last1 = tf.keras.layers.LSTM(filters, name=name+nc)
      lyr_last2 = tf.keras.layers.Bidirectional(lyr_last1, name=name+'_bidi'+nc)
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
                               name=lyr_name+"_gated_bn{}_{}_{}".format(
                                   lyr_bn, lyr_act, i+1))
        tf_x = lyr_gated(tf_x)
        if lyr_drop > 0:
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
        if lyr_drop > 0:
          tf_x = tf.keras.layers.Dropout(lyr_drop, 
                                         name=lyr_name+'_drop_{}_{}'.format(
                                             lyr_drop,i+1))(tf_x)
    return tf_x
    
    
    
  
  def setup_model(self, dict_model_config=None):   
    self.P("Initializing hyperparameters...")
    self._init_hyperparams(dict_model_config=dict_model_config)
    if self.embeddings is None:
      self._setup_word_embeddings()
    self.P("Defining model...")
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
      ker_size = col['KERNEL'] if 'KERNEL' in col.keys() else 0
      col_depth = col['DEPTH'] if 'DEPTH' in col.keys() else 0
      bn = col['BN'] if 'BN' in col.keys() else False
      end_type = col['END']
      step = col['STEP'] if 'STEP' in col.keys() else ker_size
      if self.pre_columns_end is not None:
        end_type = self.pre_columns_end
      tf_x = self._define_column(tf_input=tf_embeds,
                                 kernel=ker_size,
                                 name='C'+str(i+1),
                                 filters=n_feats,
                                 depth=col_depth,
                                 bn=bn,
                                 step=step,
                                 last_step=end_type,
                                 use_cuda=self.use_cuda
                                 )
      tf_lst_cols.append(tf_x)
    tf_x = tf.keras.layers.concatenate(tf_lst_cols)
    drp = self.dropout_end 
    tf_x = tf.keras.layers.Dropout(drp, 
                                   name='drop_{}_{}'.format(
                                       drp,0))(tf_x)
    self.n_concat_outs = len(self.model_columns) * n_feats * 2
    
    
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
      model.compile(optimizer='adam', loss='categorical_crossentropy', 
                    metrics=['acc', self.log.K_rec]
                    )
    

    elif self.model_output == 'tagging':
      ### sigmoid output
      tf_readout = tf.keras.layers.Dense(self.output_size,
                                         activation='sigmoid',
                                         name='readout_sigmoid')(tf_x)
      model = tf.keras.models.Model(inputs=tf_input,
                                    outputs=tf_readout)
      model.compile(optimizer='adam', loss='binary_crossentropy', 
                    metrics=['acc', self.log.K_rec]
                    )
    else:
      raise ValueError("Unknown model output '{}'".format(self.model_output))
    self.model = model
    self.P("Final model:\n{}".format(self.log.GetKerasModelSummary(self.model)))
    self.model_prepared = True
    return
  
  


          
      
  
  
        

if __name__ == '__main__':
  from libraries.logger import Logger
  from tagger.brain.data_loader import ALLANDataLoader
  
  cfg1 = "tagger/brain/configs/config.txt"
  
  use_raw_text = True
  save_model = True
  use_model_conversion = True
  epochs = 60
  use_loaded = True
  
  l = Logger(lib_name="ALNT",config_file=cfg1)
  l.SupressTFWarn()
  
  loader = ALLANDataLoader(log=l, multi_label=True, 
                           normalize_labels=False)
  loader.LoadData()
  
  
  valid_texts = [
      "in ce cartier aveti biro prin bucurest?",
      "cam cum sint persoanele care lucreaza la ey?",
      "si cam cat ar fi salarul pentru un junior??",
      "si cam care sunt nivelele de salarizare in finante sau contabilitate?",
      "care este salariul la inceput de cariera pentru un auditor financiar?",
      "exista posibilitatea de a putea pleca sa lucrez in alta tara?",
      "oare as putea sa lucrez si de acasa? care este politica in acest sens?",
      ]
  valid_labels = [
      ['biroul', 'bucuresti', 'zona', 'topic_sediu_bucur'],
      ['oamenii', 'topic_echip'],
      ['junior', 'topic_salar'],
      ['grilele','salariu' ,'topic_salar', 'tax', 'audit'],
      ['junior', 'topic_salar', 'tax', 'audit', 'salariu'],  
      ['internationala', 'mobilitate', 'tara', 'relocare','topic_mobil'],
      ]
  
  
  eng = ALLANTaggerCreator(log=l, 
                           dict_word2index=loader.dic_word2index,
                           dict_label2index=loader.dic_labels)
  
  eng.setup_model(dict_model_config=None) # default architecture
  
  if use_raw_text:
    eng.train_on_texts(loader.raw_documents,
                       loader.raw_labels,
                       n_epochs=epochs,
                       convert_unknown_words=use_model_conversion,
                       save=save_model,
                       skip_if_pretrained=use_loaded)
  else:
    eng.train_on_tokens(loader.x_docs, 
                        loader.y_labels,
                        n_epochs=epochs,
                        convert_unknown_words=use_model_conversion,
                        save=save_model,
                        skip_if_pretrained=use_loaded)
    
  debug_results = False
  if True:
      
    l.P("")
    dct_tags, inputs = eng.predict_text("as vrea info despre salarizare daca se poate")
    res = eng.tagdict_to_text(dct_tags)
    l.P("Result: {}".format(res))
    if debug_results:
      l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
            for x,p in zip(eng.last_labels, eng.last_probas)]))
    l.P("")
    dct_tags, inputs = eng.predict_text("unde aveti sediile mai multe in bucuresti ca as vrea sa fie aproape de casa?")
    res = eng.tagdict_to_text(dct_tags)
    l.P("Result: {}".format(res))
    if debug_results:
      l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
            for x,p in zip(eng.last_labels, eng.last_probas)]))
    
    l.P("")
    dct_tags, inputs = eng.predict_text("in ce departamente aveti pozitii deschise la acest moment in bucuresti?")
    res = eng.tagdict_to_text(dct_tags)
    l.P("Result: {}".format(res))
    if debug_results:
      l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
            for x,p in zip(eng.last_labels, eng.last_probas)]))
  
    l.P("")
    dct_tags, inputs = eng.predict_text("ce calificare este necesara pentru un post de junior contabil?")
    res = eng.tagdict_to_text(dct_tags)
    l.P("Result: {}".format(res))
    if debug_results:
      l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
            for x,p in zip(eng.last_labels, eng.last_probas)]))
    
    
    #TODO: test various scenarios !!! 
  
        
    