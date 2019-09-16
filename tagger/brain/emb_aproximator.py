# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:04:56 2019

@author: Andrei
"""
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K


from tagger.brain.base_engine import ALLANTaggerEngine

from libraries.lummetry_layers.gated import GatedDense

from time import time

class EmbeddingApproximator(ALLANTaggerEngine):
  def __init__(self, np_embeds=None, dct_w2i=None, dct_i2w=None, **kwargs):
    super().__init__(**kwargs)
    self.__name__ = 'EMBA'
    self.trained = False
    self.siamese_model = None

    if np_embeds is None:
      self._setup_word_embeddings()
      self.emb_size = self.embeddings.shape[-1]
    else:
      self.embeddings = np_embeds
      
    
    if dct_w2i is None:
      self._setup_vocabs()
    else:
      self.dic_word2index = dct_w2i
      if dct_i2w is None:
        self.dic_index2word = {v:k for k,v in dct_w2i.items()}
      else:
        self.dic_index2word = dct_i2w
    self._setup()
    return
  
  def _setup(self):
    self.embgen_model_batch_size = self.embgen_model_config['BATCH_SIZE']
    self.use_cuda = self.embgen_model_config['USE_CUDA']
    return
    
  
  def get_model(self):
    return self.model
  
  
  def _define_emb_model_layer(self, 
                              tf_inputs, 
                              layer_name,
                              layer_cfg,
                              final_layer,
                              prev_features,
                              use_cuda,
                              ):
    s_name = layer_name.lower()
    n_prev_feats = prev_features
    sequences = not final_layer
    s_type = layer_cfg['TYPE'].lower()
    b_residual = layer_cfg['RESIDUAL'] if 'RESIDUAL' in layer_cfg.keys() else False
    n_feats = layer_cfg['FEATS']
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
    elif 'conv' in s_type:
      n_ker = layer_cfg['KERNEL']
      act = layer_cfg['ACT'].lower() if 'ACT' in layer_cfg.keys() else 'relu'
      tf_x = tf.keras.layers.Conv1D(filters=n_feats,
                                    kernel_size=n_ker,
                                    strides=n_ker,
                                    name=s_name+'_conv')(tf_inputs)                                    
      tf_x = tf.keras.layers.BatchNormalization(name=s_name+'_bn')(tf_x)
      tf_x = tf.keras.layers.Activation(act, name=s_name+'_'+act)(tf_x)
      if final_layer:
        tf_x1 = tf.keras.layers.GlobalAvgPool1D(name=s_name+'_GAP')(tf_x)
        tf_x2 = tf.keras.layers.GlobalMaxPool1D(name=s_name+'_GMP')(tf_x)
        tf_x = tf.keras.layers.concatenate([tf_x1, tf_x2], name=s_name+'_concat')
    else:
      raise ValueError("Unknown '' layer type".format(s_type))
    
    if b_residual:
      if 'lstm' in s_type:
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
    if self.embgen_model_config is None:
      raise ValueError("EMB_GEN_MODEL not configured - please define dict")
    if len(self.embgen_model_config['COLUMNS']) == 0 :
      raise ValueError("EMB_GEN_MODEL columns not configured - please define columns/layers")

    if 'FINAL_DROP' in self.embgen_model_config.keys():
      drp = self.embgen_model_config['FINAL_DROP']
    else:
      drp = 0

    tf_input = tf.keras.layers.Input((None,), name='word_input')

    vocab_size = len(self.char_full_voc)
    emb_size = self.embgen_model_config['CHR_EMB_SIZE']
    tf_emb = tf.keras.layers.Embedding(vocab_size, 
                                       emb_size, 
                                       name='inp_embd')(tf_input)
    
    columns_cfg = self.embgen_model_config['COLUMNS']
    lst_columns = []
    for col_name in columns_cfg:
      column_config = columns_cfg[col_name]    
      layers_cfg = column_config['LAYERS']
      tf_x = tf_emb
      n_layers = len(layers_cfg)
      prev_features = 0
      for L in range(n_layers-1):
        layer_name = col_name+'_'+layers_cfg[L]['NAME']  
        tf_x = self._define_emb_model_layer(
                                tf_inputs=tf_x,
                                layer_name=layer_name,
                                layer_cfg=layers_cfg[L],
                                final_layer=False,
                                prev_features=prev_features,
                                use_cuda=self.use_cuda
                              )
        prev_features = layers_cfg[L]['FEATS']
      # final column end
      layer_name = col_name+'_'+layers_cfg[-1]['NAME']     
      tf_x = self._define_emb_model_layer(
                              tf_inputs=tf_x,
                              layer_name=layer_name,
                              layer_cfg=layers_cfg[-1],
                              final_layer=True,
                              prev_features=prev_features,
                              use_cuda=self.use_cuda,
                            )
      lst_columns.append(tf_x)
    
    if len(lst_columns) > 1:
      tf_x = tf.keras.layers.concatenate(lst_columns, name='concat_columns')
    else:
      tf_x = lst_columns[0]
    if drp > 0:
      tf_x = tf.keras.layers.Dropout(drp, name='drop1_{:.1f}'.format(drp))(tf_x)
    tf_x = GatedDense(units=self.emb_size*2, name='gated1')(tf_x)
    if drp > 0:
      tf_x = tf.keras.layers.Dropout(drp, name='drop2_{:.1f}'.format(drp))(tf_x)
    
    tf_x = tf.keras.layers.Dense(self.emb_size, name='emb_fc_readout')(tf_x)
    l2norm_layer = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=1), name='emb_l2_norm_readout')
    tf_readout = l2norm_layer(tf_x)
    model = tf.keras.models.Model(inputs=tf_input, outputs=tf_readout)
    model.compile(optimizer='adam', loss='logcosh')
    self.embgen_model = model
    self.embgen_model_trained = False
    self.P("Unknown words embeddings generator model:\n{}".format(
        self.log.GetKerasModelSummary(self.embgen_model)))
    return


  def _define_siamese_model(self):
    if self.embgen_model is None:
      raise ValueError("The basic model is undefined")
    tf_input1 = tf.keras.layers.Input((None,), name='inp1')  
    tf_input2 = tf.keras.layers.Input((None,), name='inp2')  
    tf_input3 = tf.keras.layers.Input((None,), name='inp3')
    
    tf_emb1 = self.embgen_model(tf_input1)
    tf_emb2 = self.embgen_model(tf_input2)
    tf_emb3 = self.embgen_model(tf_input3)
      
    triple_loss_layer = tf.keras.layers.Lambda(function=self.log.K_triplet_loss,
                                               name='triplet_loss_layer')
    
    tf_readout = triple_loss_layer([tf_emb1, tf_emb2, tf_emb3])
    
    model = tf.keras.models.Model(inputs=[tf_input1, tf_input2, tf_input3], outputs=tf_readout)  
    
    model.compile(optimizer='adam', loss=self.log.K_identity_loss)    
    self.siamese_model = model
    return model
  
  
  def _word_morph(self, word, same_caps=None):
    assert same_caps in [None, True, False]
    
    if len(word) <= 4:
      raise ValueError("Not morphing words less than 5")
    mistk_src = []
    mistk_dst = []
    
    # Manages if the mistakes can be made only from upper to upper / lower to lower or not.
    if same_caps is None and 'SAME_CAPS' in self.embgen_model_config:
      same_caps = bool(self.embgen_model_config['SAME_CAPS'])
    
    if same_caps is None:
      same_caps = False
    
    letter2letter = [chr(x) for x in range(97, 123)]
    
    if not same_caps:
      for letter in letter2letter:
        mistk_src.append(letter)
        mistk_dst.append(letter.upper())
        mistk_src.append(letter.upper())
        mistk_dst.append(letter)
    
    mistk_src += ['i','o','I','0','1','0','O','1','!','6','G','5','S','s','5','r','t']
    mistk_dst += ['1','0','1','o','I','O','0','!','1','G','6','s','5','5','S','t','r']

    mistk_src += ['7','G','E','A','1','V','T','1','l','8','B','l','I','*','-','0','9']
    mistk_dst += ['T','E','G','V','i','A','7','l','1','B','8','I','l','-','*','9','0']

    mistk_src += ['Î','ț','ă','î','Ă','Ș','ș','Ț']
    mistk_dst += ['I','t','a','i','A','S','s','T']

    mistk_src += ['I','t','a','i','A','S','s','T']
    mistk_dst += ['Î','ț','ă','î','Ă','Ș','ș','Ț']
    
    
    typo_src = ['m','s','o','i','i','i','u','r','e','c','v','n','z','z']
    typo_dst = ['n','d','p','j','o','u','y','t','r','v','b','b','s','x']
    
    typo_src += typo_dst
    typo_dst += typo_src

    for i in range(len(typo_src)):
      mistk_src.append(typo_src[i])
      mistk_dst.append(typo_dst[i])
      mistk_src.append(typo_src[i].upper())
      mistk_dst.append(typo_dst[i].upper())
      if not same_caps:
        mistk_src.append(typo_src[i].upper())
        mistk_dst.append(typo_dst[i])
        mistk_src.append(typo_src[i])
        mistk_dst.append(typo_dst[i].upper())
      #endif
    #endfor

    new_word = []
    proba = 0.5
    modded = 0
    for i,ch in enumerate(word):
      if np.random.rand() < proba:
        if np.random.rand() < 0.5 and i > 0:
          modded += 1
          proba -= 0.24
        else:
          if ch in mistk_src:
            all_ch_indexes = [i for i,x in enumerate(mistk_src) if x == ch]
            idx = np.random.choice(all_ch_indexes)
            new_ch = mistk_dst[idx]
            new_word.append(new_ch)
            modded += 1
            proba -= 0.24
          else:
            new_word.append(ch)
      else:
        new_word.append(ch)
        
    if modded == 0:
      new_word = new_word[:-1]
      modded = 1
    if len(new_word) < 4:
      new_word += ['1'] * 2
    new_word = "".join(new_word)
    return new_word
  
  def _get_siamese_datasets(self, min_word_size=4, min_nr_words=5,
                            max_word_min_count=15, force_generate=False):
    if self.dic_word2index is None:
      raise ValueError("Vocab not loaded!")
    lst_anchor = []
    lst_duplic = []
    lst_false  = []
    
    if (not force_generate) and ('DATAFILE' in self.embgen_model_config.keys()):
      fn = self.embgen_model_config['DATAFILE']
      if self.log.GetDataFile(fn) is not None:
        xa, xd, xf = self.log.LoadPickleFromData(fn)
        self._siam_data_lens = [len(x) for x in xa]
        self._siam_data_unique_lens = np.unique(self._siam_data_lens)
        return xa, xd, xf

    self.P("Generating siamese net training data from vocab")
    vlens = self.analize_vocab_and_data()
    n_words = len(self.dic_word2index)
    len_counts = np.bincount(vlens)
    for x in range(len(len_counts)-1,1, -1):
      if len_counts[x] > max_word_min_count:
        break
    max_word_size = x
    t1 = time()
    i = 0
    for word, idx in self.dic_word2index.items():
      print("\rGenerating siamese dataset {:.1f}%".format(
          (i/n_words)*100), end='', flush=True)
      i += 1
      if idx in self.SPECIALS:
        continue
      l_word = len(word)

      if  (l_word > min_word_size) and (l_word < max_word_size):
        s_duplic = self._word_morph(word)
        s_anchor = word
        i_false = (idx + np.random.randint(100,1000)) % len(self.dic_index2word)
        s_false  = self.dic_index2word[i_false]
        _len = l_word #max(len(s_duplic), len(s_anchor), len(s_false))
        s_anchor = s_anchor[:_len]
        s_duplic = s_duplic[:_len]
        s_false = s_false[:_len]

        np_anchor = np.array(self.word_to_char_tokens(s_anchor, pad_up_to=_len))
        np_duplic = np.array(self.word_to_char_tokens(s_duplic, pad_up_to=_len))
        np_false = np.array(self.word_to_char_tokens(s_false, pad_up_to=_len))
        lst_anchor.append(np_anchor)
        lst_duplic.append(np_duplic)
        lst_false.append(np_false)

    
    #generate also based on predefined list of typos or other kind of mistakes
    if 'CUSTOM_MISTAKES_FILE' in self.embgen_model_config:
      fn = self.embgen_model_config['CUSTOM_MISTAKES_FILE']
      if self.log.GetDataFile(fn) is not None:
        pairs = self.log.LoadPickleFromData(fn)
        for s_duplic, s_anchor in pairs:
          idx = self.dic_word2index[s_anchor]
          i_false = (idx + np.random.randint(100,1000)) % len(self.dic_index2word)
          s_false = self.dic_index2word[i_false]
          _len = max(len(s_anchor), len(s_duplic))
          s_anchor = s_anchor[:_len]
          s_duplic = s_duplic[:_len]
          s_false = s_false[:_len]
          s_false = s_false[:_len]
          
          np_anchor = np.array(self.word_to_char_tokens(s_anchor, pad_up_to=_len))
          np_duplic = np.array(self.word_to_char_tokens(s_duplic, pad_up_to=_len))
          np_false = np.array(self.word_to_char_tokens(s_false, pad_up_to=_len))
          lst_anchor.append(np_anchor)
          lst_duplic.append(np_duplic)
          lst_false.append(np_false)
        #endfor
      #endif
    #endif
    
    t2 = time()
    print("")
    self.P(" Done generating in {:.1f}s".format(t2-t1))    
    self._siam_data_lens = [x.size for x in lst_anchor]
    self.P("")
    self.log.ShowTextHistogram(self._siam_data_lens, 
                               caption='Siam data len distrib',
                               show_both_ends=True)
    self._siam_data_unique_lens = np.unique(self._siam_data_lens)
        
    x_anchor = np.array(lst_anchor)
    x_duplic = np.array(lst_duplic)
    x_false  = np.array(lst_false)
    self.P("Prepared siamese data with {} obs".format(x_anchor.shape[0]))
    data = x_anchor, x_duplic, x_false
    if 'DATAFILE' in self.embgen_model_config.keys():
      fn = self.embgen_model_config['DATAFILE']
      self.log.SavePickleToData(data, fn)
    return data
  
  
  def _get_siamese_generator(self, x_a, x_d, x_f):
    BATCH_SIZE = self.embgen_model_batch_size
    while True:
      for unique_len in self._siam_data_unique_lens:        
        subset_pos = self._siam_data_lens == unique_len
        np_x_a_subset = np.array(x_a[subset_pos].tolist())
        np_x_d_subset = np.array(x_d[subset_pos].tolist())
        np_x_f_subset = np.array(x_f[subset_pos].tolist())
        n_obs = np_x_a_subset.shape[0]
        n_batches = n_obs // BATCH_SIZE
        for i_batch in range(n_batches):
          b_start = (i_batch * BATCH_SIZE) % n_obs
          b_end = min(n_obs, b_start + BATCH_SIZE)          
          np_x_a_batch = np_x_a_subset[b_start:b_end]
          np_x_d_batch = np_x_d_subset[b_start:b_end]
          np_x_f_batch = np_x_f_subset[b_start:b_end]
          yield np_x_a_batch, np_x_d_batch, np_x_f_batch        
    
    
                
  
  def _get_embgen_model_generator(self, x_data):  
    BATCH_SIZE = self.embgen_model_batch_size
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
    
  
  
  
    
    
  def train_unk_words_model(self, epochs=2, approximate_embeddings=False,
                            save_embeds_every=5, force_generate=False,
                            overwrite_pretrained=False):
    """
     trains the unknown words embedding generator based on loaded embeddings
    """
    if self.embgen_model is None:
      self._define_emb_generator_model()
    if self.siamese_model is None:
      self._define_siamese_model()

    # OBSOLETE (almost)
    if approximate_embeddings: 
      min_size = 4
      # get generators
      self.get_vocab_training_data(min_size)
      gen = self._get_embgen_model_generator(self.x_data_vocab)
    # END OBSOLETE (almost)
    
    
    xa,xd,xf = self._get_siamese_datasets(force_generate=force_generate)
    self.P("Siamese data sanity check:")
    for i in range(10):
      irnd = np.random.randint(0, xa.shape[0])
      sa = self.char_tokens_to_word(xa[irnd])
      sd = self.char_tokens_to_word(xd[irnd])
      sf = self.char_tokens_to_word(xf[irnd])
      self.P(" A:{:>15}  D:{:>15}  F:{:>15}".format(sa,sd,sf))
    siam_gen = self._get_siamese_generator(xa,xd,xf)
    # fit model
    n_batches = self.embeddings.shape[0] // self.embgen_model_batch_size
    n_siam_batches = xa.shape[0] // self.embgen_model_batch_size

    avg_loss1 = []
    avg_loss2 = []
    self.P("Training EmbGen model for {} epochs".format(epochs))
    for epoch in range(epochs):
      if approximate_embeddings:
        loss1 = self._train_basic(gen, n_batches, epoch)
        avg_loss1.append(loss1)
        self.P("Epoch {} basic training done. loss:{:>7.4f}  avg:{:>7.4f}".format(
            epoch+1, loss1, np.mean(avg_loss1)))
        self.debug_unk_words_model(['creerii', 'pumul','capu','galcile'])      

      loss2 = self._train_siamese(siam_gen, n_siam_batches, epoch)
      avg_loss2.append(loss2)
      self.P("Epoch {} siam training done. loss:{:>7.4f}  avg:{:>7.4f}".format(
          epoch+1, loss2, np.mean(avg_loss2)))
      if (((epoch+1) % save_embeds_every) == 0) and epoch < (epochs-1):
        self.save_model()
        self._get_generated_embeddings()
        self.debug_unk_words_model()      
        self.P("")    
    self.save_model(overwrite_pretrained=overwrite_pretrained)                
    self._get_generated_embeddings()
    return
  
  
  def save_model(self, overwrite_pretrained=False):
    if not overwrite_pretrained:
      self.log.SaveKerasModel(self.embgen_model, label='embgen_model', use_prefix=True)
    else:
      label = self.embgen_model_config['PRETRAINED'] if 'PRETRAINED' in self.embgen_model_config.keys() else 'embgen_model'
      self.log.SaveKerasModel(self.embgen_model, label=label, use_prefix=False)
    return
    
  
  def _train_basic(self, gen, steps, epoch):
    epoch_losses = []
    n_batches = steps
    for i_batch in range(n_batches):
      x_batch, y_batch = next(gen)
      loss = self.embgen_model.train_on_batch(x_batch, y_batch)
      print("\r Basic Epoch {}: {:>5.1f}% completed [loss: {:.4f}]".format(
          epoch+1, i_batch / n_batches * 100, loss), end='', flush=True)
      epoch_losses.append(loss)
    print("\r",end="")
    epoch_loss = np.mean(epoch_losses)
    return epoch_loss
  

  def _train_siamese(self, gen, steps, epoch):
    epoch_losses = []
    n_batches = steps
    for i_batch in range(n_batches):
      x_a, x_d, x_f = next(gen)
      loss = self.siamese_model.train_on_batch([x_a, x_d, x_f])
      print("\r Siam Epoch {}: {:>5.1f}% completed [loss: {:.4f}]".format(
          epoch+1, i_batch / n_batches * 100, loss), end='', flush=True)
      epoch_losses.append(loss)
    print("\r",end="")
    epoch_loss = np.mean(epoch_losses)
    return epoch_loss
  
    
      
  def debug_unk_words_model(self, unk_words=['oferiti', 
                                             'timisoara',
                                             'bucuresti',
                                             'sediuri', 
                                             'sumt',
                                             'sumnt',
                                             'salarul',
                                             'nTimisoara',
                                             'nChisinau',
                                             'biruol',
                                             'zoma',
                                             'alariul',
                                             'ecipa',
                                             'transp',
                                             'Iqsi',
                                             'sedyuri',
                                             'adrs',
                                             'trbuie',
                                             'trb'
                                             ]):
    self.P("Testing for {} (dist='{}')".format(
                unk_words, self.dist_func_name))
    for uword in unk_words:
      if uword in self.dic_word2index.keys():
        self.P(" 'Unk' word {} found in dict at pos {}".format(
                    uword, self.dic_word2index[uword]))
        continue
      top = self.get_unk_word_similar_word(uword, top=3)
      self.P(" unk: '{}' results in: {}".format(uword, top))
    return
      
      
  def debug_known_words(self, good_words=['ochi', 'gura','gat','picior','mana','genunchi']):
    self.P("Testing known words {} (dist='{}')".format(
        good_words, self.dist_func_name))
    for word in good_words:
      idx = self.dic_word2index[word]
      orig_emb = self.embeddings[idx]
      idxs1, dist1 = self._get_closest_idx_and_distance(aprox_emb=orig_emb, top=5)
      top1 = ["'{}':{:.3f}".format(self.dic_index2word[x],y)  
              for x,y in zip(idxs1, dist1)]      
      top1 = " ".join(top1)
      self.P(" wrd: '{}' >>> embeds >>>: {}".format(word, top1))
      
      aprox_emb = self._get_approx_embed(word)
      idxs2, dist2 = self._get_closest_idx_and_distance(aprox_emb=aprox_emb, top=5,
                                                        np_embeds=self.generated_embeddings)
      top2 = ["'{}':{:.3f}".format(self.dic_index2word[x],y)  
              for x,y in zip(idxs2, dist2)]      
      top2 = " ".join(top2)
      self.P(" wrd: '{}' >>> w. embgen >>>: {}".format(word, top2))
      
    return
  
  
  def compute_performance(self, unk_words, true_words, tops=[1,3,5]):
    assert type(unk_words) == list
    assert type(true_words) == list
    assert len(unk_words) == len(true_words)
    
    self.P("Computing EmbeddingApproximator performance on {} examples ..."
           .format(len(unk_words)))
    
    max_top = max(tops)
    
    cnt_top = [0 for _ in range(len(tops))]
    nr_skipped = 0
    
    from tqdm import tqdm
    
    for i,uword in tqdm(enumerate(unk_words)):
      tword = true_words[i]
      if uword in self.dic_word2index:
#        self.P(" Unk word '{}' found in dict at pos {}".format(
#                    uword, self.dic_word2index[uword]))
        nr_skipped += 1
        continue

      top_words = self.get_unk_word_similar_word(uword, top=max_top)      
      for j,t in enumerate(tops):
        if tword in top_words[:t]:
          cnt_top[j] += 1  

    
    nr_computed = len(unk_words) - nr_skipped

    str_log = ""
    for j,t in enumerate(tops):
      _p = 100 * cnt_top[j] / nr_computed
      str_log += '\n{}/{} ({:.2f}%) @Top{}'.format(cnt_top[j], nr_computed, _p, tops[j])

    str_log += '\n{} words were skipped because they were found in dict'.format(nr_skipped)

    self.P("Results:{}".format(str_log))

    return cnt_top, nr_skipped

    
  
  
if __name__ == '__main__':
  from libraries.logger import Logger
  
  cfg1 = "tagger/brain/configs/config.txt"
  l = Logger(lib_name="EGEN",config_file=cfg1)

  eng = EmbeddingApproximator(log=l,)

  if False:
    #eng._get_siamese_datasets(min_nr_words=0)
    xa, xd, xf = eng._get_siamese_datasets(force_generate=True)
    for i in range(50):
      irnd = np.random.randint(0, xa.shape[0])
      sa = eng.char_tokens_to_word(xa[irnd])
      sd = eng.char_tokens_to_word(xd[irnd])
      sf = eng.char_tokens_to_word(xf[irnd])
      l.P(" A:{:>15}  D:{:>15}  F:{:>15}".format(sa,sd,sf))


  if True:
    eng.train_unk_words_model(epochs=100, force_generate=True, overwrite_pretrained=False)

    eng.debug_known_words()
    
  if True:
    xa, xd, _ = eng._get_siamese_datasets(force_generate=True)
    unk_words, true_words = [], []
    for i in range(xa.shape[0]):
      sa = eng.char_tokens_to_word(xa[i])
      sd = eng.char_tokens_to_word(xd[i])
      
      unk_words.append(sd)
      true_words.append(sa)

    
    unk_words  = np.array(unk_words)
    true_words = np.array(true_words) 
  
    indexes = np.random.choice(np.arange(unk_words.shape[0]),
                               20000, replace=False)
  
    res = eng.compute_performance(unk_words[indexes], true_words[indexes])