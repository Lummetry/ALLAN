# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:22:51 2019

@author: damian

"""

import json
import tensorflow as tf
import os
import numpy as np
from collections import OrderedDict

class ALLANEngine:
  """
  ALLAN 'Abstract' Engine
  """
  def __init__(self, log, DEBUG=False, MAX_CHR=100000):
    if log is None or (type(log).__name__ != 'Logger'):
      raise ValueError("Loggger object is invalid: {}".format(log))
    self.log = log
    self.config_data = self.log.config_data
    #"".join([chr(0)] + [chr(i) for i in range(32, 127)] + [chr(i) for i in range(162,256)])
    self.char_full_voc = "".join([chr(x) for x in range(MAX_CHR)])
    self.DEBUG = DEBUG
    self.min_seq_len = 20
    self.sess = None
    self.session = None
    self.trained = False
    self.prev_saved_model = []
    self.__name__ = 'ALLAN_BASE'
    self.first_run = {}
    self.frames_data = None
    self.dic_word2index = None
    self.dic_index2word = None
    self.model_ouput = None
    self.dic_labels = None
    self.dic_index2label = None
    self.embeddings = None
    self.model = None
    self.unk_words_model = None
    self.unk_words_model_trained = False
    self.emb_layer_name = 'emb_layer'
    self.startup()
    return
  
  
  def startup(self):
    self.train_config = self.config_data['TRAINING']
    self.token_config = self.config_data['TOKENS']
    self.UNK_ID = self.token_config['UNK']
    self.PAD_ID = self.token_config['PAD']
    self.SOS_ID = self.token_config['SOS']
    self.EOS_ID = self.token_config['EOS']
    self.SPECIALS = [
        self.PAD_ID,
        self.UNK_ID,
        self.SOS_ID,
        self.EOS_ID,
        ]
    self.train_folder = self.train_config['FOLDER']
    self.model_config = self.config_data['MODEL']
    self.doc_ext = self.train_config['DOCUMENT']
    self.label_ext = self.train_config['LABEL']
    self.fn_word2idx = self.config_data['WORD2IDX'] if 'WORD2IDX' in self.config_data.keys() else None
    self.fn_idx2word = self.config_data['IDX2WORD'] if 'IDX2WORD' in self.config_data.keys() else None
    self.fn_labels2idx = self.config_data['LABEL2IDX'] if 'LABEL2IDX' in self.config_data.keys() else None
    self.doc_max_words = self.config_data['DOCUMENT_MAX_WORDS']
    self.model_name = self.model_config['NAME']
    self.unk_words_model_config = self.config_data['UNK_WORDS_MODEL'] if 'UNK_WORDS_MODEL' in self.config_data.keys() else None    
    self.unk_words_model_batch_size = self.unk_words_model_config['BATCH_SIZE']
    return
        
    
  
  def shutdown(self):
    self.P("Shutdown in progress...")
    if self.sess is not None:
      self.P(" Closing tf-session...")
      self.sess.close()
      self.P(" tf-session closed.")
    if self.session is not None:
      self.P(" Closing tf-session...")
      self.sess.close()
      self.P(" tf-session closed.")
    return


  def P(self, s, t=False):    
    return self.log.P("{}: {}".format(
        self.__name__,s),show_time=t)
  
  
  def D(self, s, t=False):
    _r = -1
    if self.DEBUG:
      _r = self.log.P("[DEBUG] {}: {}".format(
                      self.__name__,s),show_time=t) 
    return _r
  
  
  def start_timer(self, tmr_id):
    self.log.start_timer(self.__name__ + '_' + tmr_id)
    return
  
  
  def end_timer(self, tmr_id):
    self.log.end_timer(self.__name__ + '_' + tmr_id)
    return

  def SaveJSON(self, json_data, fname):
    if self.output_local:
      with open(fname, 'w') as f:
        json.dump(json_data, f, sort_keys=True, indent=4)
    else:
      self.log.SaveOutputJSON(json_data, fname)


  def _run(self, _call, _output, feed_dict):    
    if (_call not in self.first_run.keys()) or (not self.first_run[_call]):
      self.first_run[_call] = False
      self.D("Call: {}  Output: {}   Input: {}".format(_call, _output, feed_dict))
    self.log.start_timer(_call)
    res = self.sess.run(_output,feed_dict=feed_dict)
    self.log.end_timer(_call,skip_first_timing=False)
    return res
  
  def word_to_char_tokens(self, word, pad_up_to=0):
    _idxs = []
    for _ch in word:
      if _ch not in self.char_full_voc:
        raise ValueError("'{}' {} not in char_vocab[{}]".format(
            _ch, ord(_ch), len(self.char_full_voc)))
      else:
        _idxs.append(self.char_full_voc.index(_ch))
    #_idxs = [self.char_full_voc.index(_ch) for _ch in word]
    n_chr = len(_idxs)
    if n_chr < pad_up_to:
      nr_added = pad_up_to - n_chr
      _idxs += [0]* (nr_added)
    return _idxs
   
  def _get_approx_embed(self, word):
    char_tokens = np.array(self.word_to_char_tokens(word)).reshape((1,-1))
    res = self.unk_words_model.predict(char_tokens)
    return res.ravel()
  
  
  def _get_closest_idx(self, aprox_emb, top=1):
    """
     get closest embedding index
    """
    assert self.embeddings is not None
    dist = ((self.embeddings - aprox_emb) ** 2).sum(axis=-1)
    _mins = np.argsort(dist)
    if top == 1:
      _min = _mins[0]
    else:
      _min = _mins[:top]
    return _min
  
  
  def get_unk_word_similar_id(self, unk_word, top=1):
    if unk_word in self.dic_word2index.keys():
      raise ValueError("'{}' is already in vocab!".format(unk_word))
    aprox_emb = self._get_approx_embed(unk_word)
    idx = self._get_closest_idx(aprox_emb=aprox_emb, top=top)
    return idx
  
  
  def get_unk_word_similar_word(self, unk_word,top=1):
    ids = self.get_unk_word_similar_id(unk_word, top=top)
    if type(ids) is np.ndarray:
      _result = [self.dic_index2word[x] for x in ids]
    else:
      _result = self.dic_index2word[ids]
    return _result
  
  
  def get_similar_words(self, word, top=1):
    idx = self.dic_word2index[word]
    embed = self.embeddings[idx]
    idxs = self._get_closest_idx(aprox_emb=embed, top=top)
    if type(idxs) is np.ndarray:
      _result = [self.dic_index2word[x] for x in idxs]
    else:
      _result = self.dic_index2word[idxs]
    return _result
  
  
  def _word_encoder(self, word, 
           convert_unknown_words=False,
           convert_to_embeddings=False,):
    idx = self.dic_word2index[word] if word in self.dic_word2index.keys() else self.UNK_ID
    if convert_to_embeddings:
      res = self.embeddings[idx]
      if convert_unknown_words and (idx == self.UNK_ID):
        res = self._get_approx_embed(word)
    else:
      if convert_unknown_words and (idx == self.UNK_ID):
        # we dont want embeds but we want to find the right word
        idx = self.get_unk_word_similar_id(word)
      res = idx
    return res
      
  
  def _get_reverse_word_dict(self):
    self.P("Constructing reverse vocab...")
    self.dic_index2word = {v:k for k,v in self.dic_word2index.items()}
  
  def _get_words(self, text):
    lst_splitted = tf.keras.preprocessing.text.text_to_word_sequence(text)
    return lst_splitted
  

  def encode(self, text, 
             text_label=None, 
             to_onehot=True,
             rank_labels=False,
             convert_unknown_words=True,
             generate_embeddings=False,
             min_len=0):
    """
    this function will tokenize or directly output the embedding represenation
    of the input list of documents together with the given labels for each
    document
    """
    s = "Starting text corpus conversion"
    if generate_embeddings:
      s += ' into embeddings'
    else:
      s += ' into tokens'
      
    if convert_unknown_words:
      s += ' and converting unknown words'
      if generate_embeddings:
        s += ' into embeddings'
      else:
        s += ' into similar tokens'
    self.P(s)
    if min_len:
      self.P("Sequences less then {} will pe padded".format(min_len))
    if type(text) in [str]:
      text = [text]
    lst_enc_texts = []
    lst_enc_labels = []
    self.last_max_size = 0
    for txt in text:
      splitted = self._get_words(txt)
      self.last_max_size = max(self.last_max_size, len(splitted))
      tokens = []
      for word in splitted:
        token = self._word_encoder(word, 
                                   convert_unknown_words=convert_unknown_words,
                                   convert_to_embeddings=generate_embeddings)
        tokens.append(token)
      if len(tokens) < min_len:
        added = min_len - len(tokens)
        if generate_embeddings:
          tokens += [self.embeddings[self.PAD_ID]] * added
        else:
          tokens += [self.PAD_ID] * added
      lst_enc_texts.append(tokens)
    if text_label is not None:
      assert type(text_label) in [list, tuple, np.ndarray], "labels must be provided as list/list or lists"
      if type(text_label[0]) in [str]:
        text_label = [text_label]
      if to_onehot:
        lst_enc_labels = self.labels_to_onehot_targets(text_label, 
                                                       rank=rank_labels)
      else:
        for lbl in text_label:
          l_labels =[self.dic_labels[x] for x in lbl]
          lst_enc_labels.append(l_labels)
    if generate_embeddings:
      lst_enc_texts = np.array(lst_enc_texts)
    if len(lst_enc_labels) > 0:
      return lst_enc_texts, lst_enc_labels
    else:
      return lst_enc_texts

    
  def decode(self, tokens, labels_idxs=None, from_onehot=True):
    """
    this function will transform a series of token sequences into text as well 
    as a list of sequences of labels indices into coresponding indices
    """
    if type(tokens[0]) in [int]:
      tokens = [tokens]
    texts = []
    labels = []
    for seq in tokens:
      txt = " ".join([self.dic_index2word[x] for x in seq])
      texts.append(txt)
    if labels_idxs is not None:
      if type(labels_idxs[0]) in [int]:
        labels_idxs = [labels_idxs]
      for seq_idxs in labels_idxs:
        if from_onehot:
          seq = np.argwhere(seq_idxs).ravel().tolist()
        else:
          seq = seq_idxs
        c_labels = [self.dic_index2labels[x] for x in seq]
        labels.append(c_labels)
    return texts, labels

  @property
  def loaded_vocab_size(self):
    return len(self.dic_word2index)  
  
  @property
  def loaded_labels_size(self):
    return len(self.dic_labels)
  
  def one_hotter(self, data):
    return tf.keras.utils.to_categorical(data, num_classes=self.output_size)
  
  def labels_to_onehot_targets(self, labels, rank=False):
    if not type(labels[0]) in [list, tuple, np.ndarray]:
      raise ValueError("labels must be provided as list of lists or 2d ndarray")
    idx_labels = []
    if type(labels[0][0]) is str:
      idx_labels = [[self.dic_labels[x] for x in obs] for obs in labels]
    else:
      idx_labels = labels
    maxes = [max(x) for x in idx_labels]
    sizes = [len(x) for x in idx_labels]
    
    if max(maxes) > 1:
      self.P("Converting labels to targets")
      lst_outs = []
      for obs in idx_labels:
        np_obs = np.array([self.one_hotter(x) for x in obs])
        np_obs = np_obs.sum(axis=0)
        if rank:
          np_obs[obs] = np.linspace(1, 0.6, num=len(obs))
        lst_outs.append(np_obs)
      np_output = np.array(lst_outs)
    elif np.unique(sizes).size != 1:
      raise ValueError("something is wrong, labels are one-hot but vector sizes differ!")
    else:
      np_output = np.array(idx_labels)
      self.P("Everything looks good, no processing required on {}".format(
          np_output.shape))
    return np_output
  
  def _predict_single(self, tokens_or_embeddings):
    """
    given a vector of tokens of embeddings (matrix) will infer the
    tags
    """
    shape = tokens_or_embeddings.shape
    #dtype = tokens_or_embeddings.dtype
    if len(shape) == 1:
      tokens_or_embeddings = tokens_or_embeddings.reshape((1,-1))
    if len(shape) == 2:
      # predict on integer tokens
      pass
    if (len(shape) == 3):
      # predict on embeds
      req_emb_size = shape[-1]
      model_inp = self.model.inputs[0].shape[-1]
      if  model_inp != req_emb_size:
        raise ValueError("Attempted to feed direct embeds obs {} in model inputs {}".format(
            shape, self.model.inputs[0]))
    x_input = tokens_or_embeddings
    if x_input.shape[1] < self.min_seq_len:
      raise ValueError("Cannot call model.predict on seq less than {} tokens".format(
          self.min_seq_len))
    preds = self.model.predict(x_input)
    return preds

  
  def predict_text(self, 
                   text, 
                   convert_unknown_words=False,
                   convert_tags=True,
                   top=5):
    """
    given a simple document will output the results based on curent model
    """
    assert self.trained and self.model is not None
    self.maybe_generate_idx2labels()
    threshold = 0.5 if "tag" in self.model_output else 0 
    self.P("Inferring '{}'".format(text))
    generate_embeddings = False
    if len(self.model.inputs[0].shape) == 3:
      generate_embeddings = True
      self.P("Model inputs {} identified to directly receive embeddings".format(
          self.model.inputs[0].shape))
    
    tokens = self.encode(text, 
                         convert_unknown_words=convert_unknown_words,
                         generate_embeddings=generate_embeddings,
                         min_len=self.min_seq_len)
    np_tokens = np.array(tokens)
    tags_probas = self._predict_single(np_tokens)
    tags_probas = tags_probas.ravel()
    tags_idxs = np.argsort(tags_probas)[::-1]
    top_idxs = tags_idxs[:top]
    top_labels = [self.dic_index2label[idx] for idx in top_idxs]
    top_prob = tags_probas[top_idxs]
    self.last_probas = top_prob
    self.last_labels = top_labels
    dct_res = OrderedDict()
    for i, idx in enumerate(top_idxs):
      if (i > 0) and tags_probas[idx] < threshold:
        # skip after first if below threshold
        continue
      if convert_tags:
        dct_res[self.dic_index2label[idx]] = tags_probas[idx]
      else:
        dct_res[idx] = tags_probas[idx]
    return dct_res
  
  def maybe_generate_idx2labels(self):
    if self.dic_index2label is None:
      self._generate_idx2labels()
    return
         
  def _generate_idx2labels(self):
    self.dic_index2label = {v:k for k,v in self.dic_labels.items()}
    return
  
  def get_stats(self, X_tokens, show=True):
    self.P("Calculating documens stats...")
    sizes = [len(seq) for seq in X_tokens]
    dict_stats = {
        "Min" : int(np.min(sizes)), 
        "Max" : int(np.max(sizes)), 
        "Avg" : int(np.mean(sizes)),
        "Med" : int(np.median(sizes)),
        }
    self.P("Done calculating documents stats.")
    if show:
      for stat in dict_stats:
        self.P(" {} docs size: {}".format(stat, dict_stats[stat]))
    return dict_stats
  
  def pad_data(self, X_tokens, max_doc_size=None):
    """
     pad data based on 'max_doc_size' or on predefined self.max_doc_size
    """
    if max_doc_size is None:
      max_doc_size = self.max_doc_size
    self.P("Padding data...")
    self.get_stats(X_tokens)
    X_data = tf.keras.preprocessing.sequence.pad_sequences(
        X_tokens, 
        value=self.PAD_ID,
        maxlen=max_doc_size, 
        padding='post', 
        truncating='post')
    self.P("Data padded to {}".format(X_data.shape))
    return X_data
    

  def _train_loop(self, X_data, y_data, batch_size, n_epochs, 
                  save_best=True,
                  save_end=True):
    """
    this is the basic 'protected' training loop loop that uses tf.keras methods and
    works both on embeddings inputs or tokenized inputs
    """
    n_obs = len(X_data)
    self.P("Training on {} obs, {} epochs, batch {}".format(
        n_obs,n_epochs, batch_size))
    n_batches = n_obs // batch_size + 1
    self.train_losses = []
    self.log.SupressTFWarn()
    best_loss = np.inf
    for epoch in range(n_epochs):
      epoch_losses = []
      for i_batch in range(n_batches):
        batch_start = (i_batch * batch_size) % n_obs
        batch_end = min(batch_start + batch_size, n_obs)
        X_batch = np.array(X_data[batch_start:batch_end])
        y_batch = np.array(y_data[batch_start:batch_end])
        loss = self.model.train_on_batch(X_batch, y_batch)
        print("\r Epoch {}: {:>5.1f}% completed [loss: {:.4f}]".format(
            epoch+1, i_batch / n_batches * 100, loss), end='', flush=True)
        self.train_losses.append(loss)
        epoch_losses.append(loss)
      print("\r",end="")
      epoch_loss = np.mean(epoch_losses)
      self.P("Epoch {} done. loss:{:>7.4f}, all avg :{:>7.4f}".format(
          epoch+1, epoch_loss,np.mean(self.train_losses)))
      if epoch_loss < best_loss:
        s_name = 'ep{}_loss{:.3f}'.format(epoch+1, epoch_loss)
        self.save_model(s_name, delete_prev_named=True)
        best_loss = epoch_loss
    self.trained = True
    self._reload_embeds_from_model()
    if save_end:
      self.save_model()
    return  
  
  
  def save_model(self, name=None, delete_prev_named=True):
    s_name = self.model_name
    if name is not None:
      s_name += '_' + name
      
    debug = not delete_prev_named
    
    if debug:      
      self.P("Saving tagger model '{}'".format(s_name))
    fn = self.log.SaveKerasModel(self.model, 
                                 s_name, 
                                 use_prefix=True,
                                 DEBUG=debug)

    if delete_prev_named:
      if self.prev_saved_model != []:
        new_list = []
        for _f in self.prev_saved_model:
          if os.path.isfile(_f):
            try:
              os.remove(_f)              
            except:
              new_list.append(_f)
        self.prev_saved_model = new_list
      self.prev_saved_model.append(fn)
    return
  
  
  def _check_model_inputs(self):
    if len(self.model.inputs[0].shape) == 3:
      self.generate_embeddings = True
      self.P("Model inputs {} identified to directly receive embeddings".format(
          self.model.inputs[0].shape))
    else:
      self.generate_embeddings = False
      self.P("Model inputs {} identified to receive tokens".format(
          self.model.inputs[0].shape))
    return
      
  
  def train_on_texts(self, 
            X_texts, 
            y_labels, 
            convert_unknown_words=True,
            batch_size=32, 
            n_epochs=1,
            force_batch=False,
            save=True,
            skip_if_pretrained=True
            ):
    """
    this methods trains the loaded/created `model` directly on text documents
    and text labels after tokenizing and (if required) converting to embeddings 
    the inputs all based on the structure of the existing `model` inputs
    """
    
    if self.model_output is None:
      raise ValueError("Model output config must be defined")
    
    if skip_if_pretrained and self.trained:
      self.P("Skipping training...")
      return
    if not (type(X_texts) in [list, tuple]):
      raise ValueError("Train function expects X_texts as a list of text documents")
      
    if not (type(y_labels) in [list, tuple]) or (type(y_labels[0]) not in [list, tuple]):
      raise ValueError("Train function expects y_labels as a list of label lists")
    n_obs = len(X_texts)
    if n_obs != len(y_labels):
      raise ValueError("X and y contain different number of observations")

    self._check_model_inputs()

    rank_labels = 'multi' in self.model_output
    
    X_tokens, y_data = self.encode(X_texts, 
                                   text_label=y_labels,
                                   to_onehot=True,
                                   rank_labels=rank_labels,
                                   convert_unknown_words=convert_unknown_words,
                                   generate_embeddings=self.generate_embeddings)
    if self.doc_max_words.lower() == 'auto':
      self.max_doc_size = self.last_max_size + 1
    else:
      self.max_doc_size = int(self.doc_max_words)

    self.P("Training on sequences of max {} words".format(self.max_doc_size))

    if force_batch:
      X_data = self.pad_data(X_tokens=X_tokens)
    else:
      batch_size = 1
      X_data = X_tokens
      self.get_stats(X_data)
      self.P("Reducing batch_size to 1 and processing doc by doc")
      
    self._train_loop(X_data, y_data, batch_size, n_epochs, 
                     save_best=save, save_end=save)
    return



  def train_on_tokens(self, 
                      X_tokens, 
                      y_labels,
                      batch_size=32, 
                      n_epochs=1,
                      force_batch=False,
                      save=True,
                      skip_if_pretrained=False):
    """
    this method assumes a `model` has been created and it accepts
    sequences of tokens as inputs. y_labels are indices of labels
    """
    if self.model_ouput is None:
      raise ValueError("Model output config must be defined")
      
    if skip_if_pretrained and self.trained:
      self.P("Skipping training...")
    if not (type(X_tokens) in [list, tuple, np.ndarray]):
      raise ValueError("Train function expects X_texts as a ndarray or list-of-lists of tokens")
      
    if type(y_labels) not in [list, tuple, np.ndarray]:
      raise ValueError("Train function expects y_labels as a ndarray or list of label lists")
      
    n_obs = len(X_tokens)
    if n_obs != len(y_labels):
      raise ValueError("X and y contain different number of observations")
    
    rank_labels = 'multi' in self.model_output
    
    y_data = self.labels_to_onehot_targets(y_labels, rank=rank_labels)

    if self.doc_max_words.lower() == 'auto':
      self.max_doc_size = max([len(s) for s in X_tokens]) + 1
    else:
      self.max_doc_size = int(self.doc_max_words)
    
    self.P("Training on sequences of max {} words".format(self.max_doc_size))

    # TODO: must implement embedding generation for proposed tokenized  data
    self._check_model_inputs()
    ###
    
    if force_batch:
      X_data = self.pad_data(X_tokens=X_tokens)
    else:
      batch_size = 1
      X_data = X_tokens
      self.P("Reducing batch_size to 1 and processing doc by doc")

    self._train_loop(X_data, y_data, batch_size, n_epochs, 
                     save_best=save, save_end=save)
    return
  
  
  def _reload_embeds_from_model(self,):
    self.P("Reloading embeddings from model")
    if self.generate_embeddings:
      self.P("Cannot reload embeddings: input size {}. second layer: {}".format(
          self.model.inputs[0].shape, self.model.layers[1].__class__.__name__))
      return
    lyr_emb = None
    for lyr in self.model.layers:
      if lyr.name == self.emb_layer_name:
        lyr_emb = lyr
    if lyr_emb is None:
      raise ValueError("Embeddings layer not found!")
    self.embeddings = lyr_emb.get_weights()[0]
    self.P("Embeddings reloaded from model {}".format(
        self.embeddings.shape))
  
  
if __name__ == '__main__':
  pass