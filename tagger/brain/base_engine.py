# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:22:51 2019

@author: damian



TODO:
  - add EmbeddingEncoderModel as alternative to embedding-lookup and similarity model
  

"""

import tensorflow as tf
import os
import numpy as np
from collections import OrderedDict

from libraries.generic_obj import LummetryObject
from libraries.lummetry_layers.gated import GatedDense
from libraries.logger import Logger
from time import time

__VER__ = '0.9.1'

class ALLANTaggerEngine(LummetryObject):
  """
  ALLAN 'Abstract' Engine
  """
  def __init__(self, log: Logger, 
               dict_word2index=None,
               dict_label2index=None,
               output_size=None,
               vocab_size=None,
               embed_size=None,
               DEBUG=False, MAX_CHR=100000,
               TOP_TAGS=None):
    if log is None or (type(log).__name__ != 'Logger'):
      raise ValueError("Loggger object is invalid: {}".format(log))
    #"".join([chr(0)] + [chr(i) for i in range(32, 127)] + [chr(i) for i in range(162,256)])
    self.MAX_CHR = MAX_CHR
    self.DEBUG = DEBUG
    self.min_seq_len = 20
    self.sess = None
    self.session = None
    self.trained = False
    self.pre_inputs = None
    self.pre_outputs = None
    self.pre_columns_end = None
    self.TOP_TAGS = TOP_TAGS
    self.prev_saved_model = []
    self.first_run = {}
    self.frames_data = None
    self.model_ouput = None
    self.embeddings = None
    self.generated_embeddings = None
    self.model = None
    self.embgen_model = None
    self.x_data_vocab = None
    self.output_size = len(dict_label2index) if dict_label2index is not None else output_size
    self.vocab_size = len(dict_word2index) if dict_word2index is not None else vocab_size
    self.dic_word2index = dict_word2index
    self.dic_labels = dict_label2index
    self.embed_size = embed_size
    self.emb_layer_name = 'emb_layer'
    super().__init__(log=log, DEBUG=DEBUG)
    return
  
  
  
  
  
  def startup(self):
    super().startup()
    self.__name__ = 'AT_TE'
    self.version = __VER__
    self.P("Init ALLANEngine v{}...".format(self.version))
    self.char_full_voc = "".join([chr(x) for x in range(self.MAX_CHR)])
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
    self.embgen_model_config = self.config_data['EMB_GEN_MODEL'] if 'EMB_GEN_MODEL' in self.config_data.keys() else None    
    self.model_config = self.config_data['MODEL']
    self.doc_ext = self.train_config['DOCUMENT']
    self.label_ext = self.train_config['LABEL']
    if self.TOP_TAGS is None:
      self.TOP_TAGS = self.config_data['TOP_TAGS'] if 'TOP_TAGS' in self.config_data.keys() else 10
    self.fn_word2idx = self.config_data['WORD2IDX'] if 'WORD2IDX' in self.config_data.keys() else None
    self.fn_idx2word = self.config_data['IDX2WORD'] if 'IDX2WORD' in self.config_data.keys() else None
    self.fn_labels2idx = self.config_data['LABEL2IDX'] if 'LABEL2IDX' in self.config_data.keys() else None
    self.doc_size = self.model_config['DOC_SIZE']
    self.model_name = self.model_config['NAME']
    self.dist_func_name = self.config_data['DIST_FUNC']
    if self.dic_word2index is not None:
      self._get_reverse_word_dict()
      self._get_vocab_stats()    
    self._generate_idx2labels()
    return
        
      
  def _setup_vocabs_and_dicts(self):
    self.P("Loading labels file '{}'".format(self.fn_labels2idx))
    if ".txt" in self.fn_labels2idx:
      dict_labels2idx = self.log.LoadDictFromModels(self.fn_labels2idx)
    else:
      dict_labels2idx = self.log.LoadPickleFromModels(self.fn_labels2idx)      
    if dict_labels2idx is None:
      raise ValueError(" No labels2idx dict found")
    dic_index2label = {v:k for k,v in dict_labels2idx.items()}
    self.dic_labels = dict_labels2idx
    self.dic_index2label = dic_index2label
    self._setup_vocabs(self.fn_word2idx, self.fn_idx2word)
    return
    
  
  def _setup_word_embeddings(self, embeds_filename=None):
    self.embeddings = None
    fn_emb = embeds_filename
    if fn_emb is None:
      fn_emb = self.model_config['EMBED_FILE'] if 'EMBED_FILE' in self.model_config.keys() else ""
      fn_emb = self.log.GetModelFile(fn_emb)
    if os.path.isfile(fn_emb):
      self.P("Loading embeddings {}...".format(fn_emb[-25:]))
      self.embeddings = np.load(fn_emb, allow_pickle=True)
      self.P(" Loaded embeddings: {}".format(self.embeddings.shape))    
      self.emb_size = self.embeddings.shape[-1]
      self.vocab_size = self.embeddings.shape[-2]
    else:
      self.P("WARNING: Embed file '{}' does not exists. embeddings='None'".format(
          fn_emb))
      if self.emb_size == 0:
        raise ValueError("No `EMBED_SIZE` defined in config and embed file could not be loaded!")
    return  
  
  
  def _setup_similarity_embeddings(self, embeds_filename=None):
    self.generated_embeddings = None
    fn_emb = embeds_filename
    if fn_emb is None:
      fn_emb = self.embgen_model_config['EMBED_FILE'] if 'EMBED_FILE' in self.embgen_model_config.keys() else ""
    if self.log.GetModelsFile(fn_emb) is not None:
      self.P("Loading similarity embeddings {}...".format(fn_emb))
      self.generated_embeddings = self.log.LoadPickleFromModels(fn_emb)
      self.P(" Loaded similarity embeddings: {}".format(self.generated_embeddings.shape))    
    else:
      self.P("WARNING: Embed file '{}' does not exists. generated_embeddings='None'".format(
          fn_emb))
    return  


  def _init_hyperparams(self, dict_model_config=None):
    if dict_model_config is not None:
      self.model_config = dict_model_config    
      self.P("Using external model parameters")
      
    self.seq_len = self.model_config['SEQ_LEN'] if 'SEQ_LEN' in self.model_config.keys() else None
    if self.seq_len == 0:
      self.seq_len = None
    self.emb_size = self.model_config['EMBED_SIZE'] if 'EMBED_SIZE' in self.model_config.keys() else 0
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
  
  
  def _get_generated_embeddings(self, x_data_vocab=None):
    if self.embgen_model is None:
      raise ValueError("`embgen_model` must be trained before generating embeddings")
    self.P("Inferring generated embeddings with `embgen_model`...")
    if x_data_vocab is None:
      if self.x_data_vocab is None:        
        x_data_vocab = self._convert_vocab_to_training_data()
      else:
        x_data_vocab = self.x_data_vocab
      
    np_embs = np.zeros((self.embeddings.shape), dtype=np.float32)
    lens = np.array([len(x) for x in self.x_data_vocab])
    unique_lens = np.unique(lens)
    t1 = time()
    iters = len(unique_lens)
    for i,unique_len in enumerate(unique_lens):
      print("\rInferring generated embeddings: {:.1f}%".format(
          ((i+1)/iters)*100), end='', flush=True)
      mask = lens == unique_len
      batch = self.x_data_vocab[mask].tolist()
      np_batch = np.array(batch)
      yhat = self.embgen_model.predict(np_batch)
      np_embs[mask]  = yhat
    print("\r",end='')
    self.generated_embeddings = np_embs
    if self.embeddings.shape != self.generated_embeddings.shape:
      raise ValueError("Embs {} differ from generated ones {}".format(
          self.embeddings.shape, self.generated_embeddings.shape))
    t2 = time()
    self.P("Done inferring generated embeddings in {:.1f}s.".format(t2-t1))
    fn = self.embgen_model_config['EMBED_FILE'] if 'EMBED_FILE' in self.embgen_model_config.keys() else "embgen_embeds.pkl"
    self.log.SavePickleToModels(self.generated_embeddings, fn)
    return 
  
  
  def analize_vocab_and_data(self, compute_lens=False):
    self.P("Analyzing given vocabulary:")
    voc_lens = [len(self.dic_index2word[x]) for x in range(len(self.dic_index2word))]
    self.log.ShowTextHistogram(voc_lens, 
                               caption='Vocabulary word len distrib',
                               show_both_ends=True)
    if self.x_data_vocab is not None:
      data_lens = [len(x) for x in self.x_data_vocab]
      self.log.ShowTextHistogram(data_lens, 
                                 caption='Vocab-based {} obs'.format(len(data_lens)),
                                 show_both_ends=True)
      if compute_lens:
        self._vocab_lens = np.array(data_lens)
        self._unique_vocab_lens = np.unique(data_lens)
    else:
      self.P("x_data_vocab` is `none`")
    return voc_lens
    

  def _convert_vocab_to_training_data(self, min_word_size=5):
    if self.x_data_vocab is not None:
      self.P("WARNING: `x_data_vocab` already is loaded")
    self.P("Converting vocabulary to training data...")
    self.P(" Post-processing with min_word_size={}:".format(min_word_size))
    t1 = time()
    x_data = []
    for i_word in range(self.embeddings.shape[0]):
      if i_word in self.SPECIALS:
        x_data.append([i_word] + [self.PAD_ID]* min_word_size)
        continue
      else:
        x_data.append(self.word_to_char_tokens(self.dic_index2word[i_word], 
                                       pad_up_to=min_word_size))
    self.x_data_vocab = np.array(x_data)
    self.analize_vocab_and_data(compute_lens=True)
    self.P(" Training data unique lens: {}".format(self._unique_vocab_lens))
    t2 = time()
    self.P("Done generating vocab training data in {:.1f}s.".format(t2-t1))
    return self.x_data_vocab
  
  
  def get_vocab_training_data(self, min_word_size=5):
    self._convert_vocab_to_training_data(
                              min_word_size=min_word_size)
    return
    

  
  def _setup_vocabs(self, fn_words_dict=None, fn_idx_dict=None):
    if fn_words_dict is None:
      fn_words_dict = self.fn_word2idx
    if fn_idx_dict is None:
      fn_idx_dict = self.fn_idx2word
      
    self.P("Loading vocabs...")
    if ".txt" in fn_words_dict:
      dict_word2idx = self.log.LoadDictFromModels(fn_words_dict)
    else:
      dict_word2idx = self.log.LoadPickleFromModels(fn_words_dict)
    if dict_word2idx is None:
      self.P("  No word2idx dict found")
    else:
      self.P(" Found word2idx[{}]".format(len(dict_word2idx)))

    if ".txt" in fn_idx_dict:
      dict_idx2word = self.log.LoadDictFromModels(fn_idx_dict)
    else:
      dict_idx2word = self.log.LoadPickleFromModels(fn_idx_dict)
      if type(dict_idx2word) in [list, tuple]:
        dict_idx2word = {i:v for i,v in enumerate(dict_idx2word)}
    if dict_idx2word is None:
      self.P(" No idx2word dict found")
    else:
      self.P(" Found idx2word[{}]".format(len(dict_idx2word)))
      
    if (dict_word2idx is None) and (dict_idx2word is not None):
      dict_word2idx = {v:k for k,v in dict_idx2word.items()}
      
    if (dict_word2idx is not None) and (dict_idx2word is None):
      dict_idx2word = {v:k for k,v in dict_word2idx.items()}
      
    self.dic_word2index = dict_word2idx
    self.dic_index2word = dict_idx2word
    return
  
  
  
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
  
  def char_tokens_to_word(self, tokens):
    chars = [self.char_full_voc[x] for x in tokens if x != 0]
    return "".join(chars)
  
  def _setup_dist_func(self, func_name='cos'):
    if func_name == 'l2':
      func = lambda x,y: ((x-y)**2).sum(axis=-1)
    elif func_name == 'l1':
      func = lambda x,y: np.abs(x-y).sum(axis=-1)
    elif func_name == 'cos':
      func = lambda x,y: 1 - (x.dot(y) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y)))
    else:
      raise ValueError("Unknown distance function '{}'".format(func_name))
    return func
  
  def dist(self, target, source=None):
    if len(target.shape) > 1:
      raise ValueError("Target must be a emb vector. Received {}".format(
          target.shape))
    if source is None:
      source = self.embeddings
    f = self._setup_dist_func(self.dist_func_name)
    return f(source, target)
      
  
  def _get_approx_embed(self, word):
    return self.__get_approx_embed(word)
  
  def __get_approx_embed(self, word):
    """
    INITIAL APPROACH WAS NOT RELIABLE:
        1. get aprox embed via regression model
        2.1. calculate closest real embedding -> id 
          or
        2.2. send the embed directly to the mode
    
    CORRECT (CURRENT) APPROACH IS TO: 
      determine closest word based on second mebedding matrix (similarity word matrix)
        
    """
    char_tokens = np.array(self.word_to_char_tokens(word, pad_up_to=5)).reshape((1,-1))
    res = self.embgen_model.predict(char_tokens)
    return res.ravel()
  
  
  def _get_closest_idx(self, aprox_emb, top=1, np_embeds=None):
    """
     get closest embedding index
    """
    if  (self.embeddings is None) and (np_embeds is None):
      raise ValueError("Both emb matrices are none!")
    
    if np_embeds is None:
      np_embeds = self.embeddings
      
    dist = self.dist(target=aprox_emb, source=np_embeds)
    _mins = np.argsort(dist)
    if top == 1:
      _min = _mins[0]
    else:
      _min = _mins[:top]
    return _min
  
  
  def _get_closest_idx_and_distance(self, aprox_emb, top=1, np_embeds=None):
    """
     get closest embedding index
    """
    if  (self.embeddings is None) and (np_embeds is None):
      raise ValueError("Both emb matrices are none!")
    
    if np_embeds is None:
      np_embeds = self.embeddings

    dist = self.dist(target=aprox_emb, source=np_embeds)
    _mins = np.argsort(dist)
    _dist = dist[_mins]
    if top == 1:
      _min = _mins[0]
      _dst = _dist[0]
    else:
      _min = _mins[:top]
      _dst = _dist[:top]      
    return _min, _dst
  
  
  def _get_token_from_embed(self, np_embed):
    if self.embeddings is None:
      raise ValueError("Embeddings matrix is undefined!")
    matches = (self.embeddings == np_embed).sum(axis=-1) == len(np_embed)
    if np.any(matches):
      return np.argmax(matches)
    else:
      return -1
    
  def _get_tokens_from_embeddings(self, np_embeddings):
    tokens = []
    for np_embed in np_embeddings:
      tokens.append(self._get_token_from_embed(np_embed))
    return tokens
  
  
  def get_unk_word_similar_id(self, unk_word, top=1):
    if unk_word in self.dic_word2index.keys():
      self.P("WARNING: presumed '{}' unk word is already in vocab!".format(unk_word))
    if self.generated_embeddings is None:
      raise ValueError("`generated_embeddings` matrix must be initialized before calculating unk word similarity")
    # first compute generated embedding
    aprox_emb = self.__get_approx_embed(unk_word)
    # get closest words id from the generated embeddings the ids will be the same in the real embeddings matrix
    idx = self._get_closest_idx(aprox_emb=aprox_emb, top=top, np_embeds=self.generated_embeddings)
    return idx
  
  
  def get_unk_word_similar_word(self, unk_word, top=1):
    ids = self.get_unk_word_similar_id(unk_word, top=top)
    if type(ids) is np.ndarray:
      _result = [self.dic_index2word[x] for x in ids]
    else:
      _result = self.dic_index2word[ids]
    return _result
  
  
  def get_similar_words_by_text(self, word, top=1):
    idx = self.dic_word2index[word]
    embed = self.embeddings[idx]
    idxs = self._get_closest_idx(aprox_emb=embed, top=top)
    if type(idxs) is np.ndarray:
      _result = [self.dic_index2word[x] for x in idxs]
    else:
      _result = self.dic_index2word[idxs]
    return _result
  

  def get_similar_words_by_id(self, id_word, top=1):
    embed = self.embeddings[id_word]
    idxs = self._get_closest_idx(aprox_emb=embed, top=top)
    if type(idxs) is np.ndarray:
      _result = [self.dic_index2word[x] for x in idxs]
    else:
      _result = self.dic_index2word[idxs]
    return _result

  
  def _word_encoder(self, word, convert_unknown_words=False,):
    if self.embeddings is None:
      self._setup_word_embeddings()
      if self.embeddings is None:
        raise ValueError("Embeddings loading failed!")
    idx = self.dic_word2index[word] if word in self.dic_word2index.keys() else self.UNK_ID
    if convert_unknown_words and (idx == self.UNK_ID):
      idx = self.get_unk_word_similar_id(word)
    if idx in self.SPECIALS:
      idx = self.UNK_ID      
    emb = self.embeddings[idx]
    return idx, emb
      
  
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
             direct_embeddings=False,
             fixed_len=0,
             DEBUG=False):
    """
    this function will tokenize or directly output the embedding represenation
    of the input list of documents together with the given labels for each
    document
    """
    s = "Starting text corpus conversion"
    if direct_embeddings:
      s += ' into embeddings'
    else:
      s += ' into tokens'
      
    if convert_unknown_words:
      s += ' and converting unknown words'
      if direct_embeddings:
        s += ' into embeddings'
      else:
        s += ' into similar tokens'
    if DEBUG:
      self.P(s)
    if fixed_len and DEBUG:
      self.P("Sequences less then {} will pe padded and those above will be truncated".format(fixed_len))
    if type(text) in [str]:
      text = [text]
    lst_enc_texts = []
    lst_enc_labels = []
    self.last_max_size = 0
    nr_obs = len(text)
    for i,txt in enumerate(text):
      if not DEBUG and (len(text) > 10):
        print("\rProcessing {:.1f}% of input documents...".format(
            i/nr_obs * 100), end='', flush=True)
      splitted = self._get_words(txt)
      self.last_max_size = max(self.last_max_size, len(splitted))
      tkns = []
      embs = []
      for word in splitted:
        tk,em = self._word_encoder(word, 
                                   convert_unknown_words=convert_unknown_words,
                                   )
        tkns.append(tk)
        embs.append(em)

      if direct_embeddings:
        tokens = embs
      else:
        tokens = tkns
      if DEBUG:
        self.P("Converted:")
        self.P("  '{}'".format(text))
        self.P(" into")
        self.P("  '{}'".format(self.decode(tkns))) 
      if len(tokens) < fixed_len:
        added = fixed_len - len(tokens)
        if direct_embeddings:
          tokens += [self.embeddings[self.PAD_ID]] * added
        else:
          tokens += [self.PAD_ID] * added
      if fixed_len > 0:
        tokens = tokens[:fixed_len]
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
    if direct_embeddings:
      lst_enc_texts = np.array(lst_enc_texts)
    if len(lst_enc_labels) > 0:
      return lst_enc_texts, lst_enc_labels
    else:
      return lst_enc_texts

    
  def decode(self, tokens, 
             tokens_as_embeddings=False,
             labels_idxs=None, 
             labels_from_onehot=True):
    """
    this function will transform a series of token sequences into text as well 
    as a list of sequences of labels indices into coresponding indices
    """
    if (("int" in str(type(tokens[0]))) or 
        (type(tokens) == np.ndarray and len(tokens.shape) == 2)):
      tokens = [tokens]
    texts = []
    labels = []
    for seq in tokens:
      if tokens_as_embeddings:
        seq = self._get_tokens_from_embeddings(seq)
      txt = " ".join([self.dic_index2word[x] for x in seq if x != self.PAD_ID]) 
      texts.append(txt)
    if labels_idxs is not None:
      if type(labels_idxs[0]) in [int]:
        labels_idxs = [labels_idxs]
      for seq_idxs in labels_idxs:
        if labels_from_onehot:
          seq = np.argwhere(seq_idxs).ravel().tolist()
        else:
          seq = seq_idxs
        c_labels = [self.dic_index2labels[x] for x in seq]
        labels.append(c_labels)
      return texts, labels
    return texts

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
                   convert_unknown_words=True,
                   convert_tags=True,
                   top=None,
                   return_input_processed=True,
                   force_below_threshold=True,
                   DEBUG=False,
                   verbose=1):
    """
    given a simple document will output the results based on curent model
      Args:
        text : the document that can be one string or a list of strings
        convert_unknown_words : True will use siamse net to find unk words
        convert_tags : True will convert tag-id into tag names
        top : number of top findings (5)
      
      Returns:
        the found tags dict in {tag: proba ...} format
    """
    if top is None:
      top = self.TOP_TAGS
    assert self.trained and self.model is not None
    self.maybe_generate_idx2labels()
    if DEBUG: 
      self.P("Inferring initial text '{}'".format(text))
    direct_embeddings = False
    if len(self.model.inputs[0].shape) == 3:
      direct_embeddings = True
      if DEBUG:
        self.P("Model inputs {} identified to directly receive embeddings".format(
            self.model.inputs[0].shape))
    
    tokens = self.encode(text, 
                         convert_unknown_words=convert_unknown_words,
                         direct_embeddings=direct_embeddings,
                         fixed_len=self.doc_size,
                         DEBUG=DEBUG)
    processed_input = self.decode(tokens=tokens, tokens_as_embeddings=direct_embeddings)[0]
    if verbose >= 1:
      self.P("Inferring (decoded): '{}'".format(processed_input))
    np_tokens = np.array(tokens)
    np_tags_probas = self._predict_single(np_tokens)
    tags = self.array_to_tags(np_tags_probas, 
                              top=top, 
                              convert_tags=convert_tags,
                              force_below_threshold=force_below_threshold)
    if DEBUG:
      top_10_preds = self.array_to_tags(
                                        np_tags_probas, 
                                        top=10, 
                                        convert_tags=True,
                                        force_below_threshold=True)
      self.P("  Predicted: {}".format("".join(["'{}':{:.3f} ".format(k,v) 
                                    for k,v in top_10_preds.items()])))
    if return_input_processed:
      return tags, (text, processed_input)
    else:
      return tags
  
  
  def array_to_tags(self, np_probas, top=5, convert_tags=True, force_below_threshold=False):
    threshold = 0.5 if "tag" in self.model_output else 0
    np_probas = np_probas.ravel()
    tags_idxs = np.argsort(np_probas)[::-1]
    top_idxs = tags_idxs[:top]
    top_labels = [self.dic_index2label[idx] for idx in top_idxs]
    top_prob = np_probas[top_idxs]
    self.last_probas = top_prob
    self.last_labels = top_labels
    dct_res = OrderedDict()
    for i, idx in enumerate(top_idxs):
      if not force_below_threshold:
        if (i > 0) and (np_probas[idx] < threshold):
          # skip after first if below threshold
          continue
      if convert_tags:
        dct_res[self.dic_index2label[idx]] = float(np_probas[idx])
      else:
        dct_res[idx] = float(np_probas[idx])
    return dct_res
  
  def maybe_generate_idx2labels(self):
    if self.dic_index2label is None:
      self._generate_idx2labels()
    return
         
  def _generate_idx2labels(self):
    if self.dic_labels is not None:
      self.dic_index2label = {v:k for k,v in self.dic_labels.items()}
    else:
      self.dic_index2label = None
    return
  
  def get_stats(self, X_tokens, show=True, X_docs=None, X_labels=None):
    self.P("Calculating documens stats...")
    sizes = [len(seq) for seq in X_tokens]
    idxs_min = np.argsort(sizes)
    dict_stats = {
        "Min" : int(np.min(sizes)), 
        "Max" : int(np.max(sizes)), 
        "Avg" : int(np.mean(sizes)),
        "Med" : int(np.median(sizes)),
        }
    self.P("Done calculating documents stats.")
    if show:
      for stat in dict_stats:
        self.P("  {} docs size: {}".format(stat, dict_stats[stat]))
      self.P("  Example of small docs:")
      for i in range(5):
        idx = idxs_min[i]
        i_sz = sizes[idx]
        if X_docs is not None:
          s_doc = X_docs[idx]
        else:
          s_doc = self.decode(X_tokens[idx], 
                              tokens_as_embeddings=self.direct_embeddings)
        lbl = ''
        if X_labels is not None:
          lbl = 'Label: {}'.format(X_labels[idx])
        self.P("    ID:{:>4} Size:{:>2}  Doc: '{}'  {}".format(
            idx, i_sz, s_doc, lbl))
    return dict_stats
  
  def __pad_data(self, X_tokens, max_doc_size=None):
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
                  X_text_valid=None, y_text_valid=None,
                  save_best=True,
                  save_end=True, 
                  test_every_epochs=1,
                  DEBUG=True):
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
    self.train_recall_history = []
    for epoch in range(n_epochs):
      epoch_losses = []
      for i_batch in range(n_batches):
        batch_start = (i_batch * batch_size) % n_obs
        batch_end = min(batch_start + batch_size, n_obs)
        X_batch = np.array(X_data[batch_start:batch_end].tolist())
        y_batch = np.array(y_data[batch_start:batch_end])
        batch_output = self.model.train_on_batch(X_batch, y_batch)
        s_bout = self.log.EvaluateSummary(self.model, batch_output)
        loss = batch_output[0] if type(batch_output)  in [list, np.ndarray, tuple] else batch_output
        batch_info = "Epoch {:>3}: {:>5.1f}% completed [{}]".format(
            epoch+1, i_batch / n_batches * 100, s_bout)        
        print("\r {}".format(batch_info), end='', flush=True)
        self.train_losses.append(loss)
        epoch_losses.append(loss)        
        self.trained = True
      print("\r",end="")
      epoch_loss = np.mean(epoch_losses)
      self.P("Epoch {} done. loss:{:>7.4f}, all avg :{:>7.4f}, last batch: [{}]".format(
          epoch+1, epoch_loss,np.mean(self.train_losses), s_bout))
      if (epoch > 0) and (test_every_epochs > 0) and (X_text_valid is not None) and ((epoch+1) % test_every_epochs == 0):
        self.P("Testing on epoch {}".format(epoch+1))
        self.test_model_on_texts(lst_docs=X_text_valid, lst_labels=y_text_valid,
                                 DEBUG=True)
      if epoch_loss < best_loss:
        s_name = 'ep{}_loss{:.3f}'.format(epoch+1, epoch_loss)
        self.save_model(s_name, delete_prev_named=True)
        best_loss = epoch_loss
    self.P("Model training done.")
    self.P("Train recall history: {}".format(self.train_recall_history))
    self._reload_embeds_from_model()
    if save_end:
      self.save_model()    
    return  
  
  
  def save_model(self, name=None, delete_prev_named=False, DEBUG=False):
    s_name = self.model_name
    if name is not None:
      s_name += '_' + name
      
    debug = (not delete_prev_named) or DEBUG
    
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
      self.direct_embeddings = True
      self.P("Model inputs {} identified to directly receive embeddings".format(
          self.model.inputs[0].shape))
    else:
      self.direct_embeddings = False
      self.P("Model inputs {} identified to receive tokens".format(
          self.model.inputs[0].shape))
    return
      
  
  def train_on_texts(self, 
            X_texts, 
            y_labels, 
            X_texts_valid=None,
            y_labels_valid=None,
            convert_unknown_words=True,
            batch_size=32, 
            n_epochs=1,
            save=True,
            skip_if_pretrained=True, 
            test_every_epochs=5,
            DEBUG=True,            
            ):
    """
    this methods trains the loaded/created `model` directly on text documents
    and text labels after tokenizing and (if required) converting to embeddings 
    the inputs all based on the structure of the existing `model` inputs
    """
    
    if self.model is None:
      raise ValueError("Model is undefined!")
    
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
    
    if convert_unknown_words and self.generated_embeddings is None:
      self.setup_embgen_model()
      

    rank_labels = 'multi' in self.model_output
    
    
    pad_data = self.doc_size
    
    X_tokens, y_data = self.encode(X_texts, 
                                   text_label=y_labels,
                                   to_onehot=True,
                                   rank_labels=rank_labels,
                                   convert_unknown_words=convert_unknown_words,
                                   direct_embeddings=self.direct_embeddings,
                                   fixed_len=pad_data)

    self.max_doc_size = self.doc_size

    self.P("Training on sequences of max {} words".format(self.max_doc_size))

    if pad_data > 0:
      X_data = X_tokens
    else:
      batch_size = 1
      X_data = X_tokens
      self.get_stats(X_data, X_labels=y_labels)
      self.P("Reducing batch_size to 1 and processing doc by doc")

    idxs_chk = [133] + np.random.choice(n_obs, size=5, replace=False).tolist()
    if X_texts_valid is None:
      X_texts_valid = [X_texts[x] for x in idxs_chk]
      y_labels_valid = [y_labels[x] for x in idxs_chk]
      
    if self.direct_embeddings:
      self.P("Sanity check before training loop for direct embeddings:")
      for idx in idxs_chk:
        x_e = X_data[idx]
        y_l = y_data[idx]
        txt = X_texts[idx]
        lbl = y_labels[idx]
        x_txt = self.decode(tokens=x_e, tokens_as_embeddings=True)
        y_lbl = self.array_to_tags(y_l)
        self.P("  Doc: '{}'".format(txt))
        self.P("  DEC: '{}'".format(x_txt[0]))
        self.P("  Lbl:  {}".format(lbl))
        self.P("  DEC:  {}".format(y_lbl))
        self.P("")

    
    self._train_loop(X_data, y_data, batch_size, n_epochs, 
                     X_text_valid=X_texts_valid, y_text_valid=y_labels_valid,
                     save_best=save, save_end=save, test_every_epochs=test_every_epochs)
    return self.train_recall_history



  def __train_on_tokens(self, 
                      X_tokens, 
                      y_labels,
                      batch_size=32, 
                      n_epochs=1,
                      save=True,
                      skip_if_pretrained=False):
    """
    TODO: Check this one and make it public!
    
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

    self.max_doc_size = self.doc_size
    
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
    self._check_model_inputs()
    if self.direct_embeddings:
      self.P("Skip reload: Cannot reload embeddings (input is: {}, second layer: {}".format(
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
    
  def maybe_load_pretrained(self):
    _res = False
    if "PRETRAINED" in self.model_config.keys():
      fn = self.model_config['PRETRAINED']
      _ver = ''
      _f = 0
      for x in fn:
        if x.isdigit():
          _ver += x
        if x == '_':
          if _f == 0:
            _ver += "."
            _f += 1
          else:
            break
      self.version += '.' + _ver
      if self.log.GetModelsFile(fn) is not None:
        self.P("Loading pretrained model {}".format(fn))
        self.model = self.log.LoadKerasModel(
                                  fn,
                                  custom_objects={
                                      "GatedDense" : GatedDense,
                                      "K_rec" : self.log.K_rec,
                                      })
        _res = True
        self._reload_embeds_from_model()
    return _res
  
  
  def maybe_load_pretrained_embgen(self):
    _res = False
    if "PRETRAINED" in self.embgen_model_config.keys():
      fn = self.embgen_model_config['PRETRAINED']
      if self.log.GetModelsFile(fn) is not None:
        self.P("Loading pretrained embgen model {}".format(fn))
        self.embgen_model = self.log.LoadKerasModel(
                                    fn,
                                    custom_objects={
                                      "GatedDense" : GatedDense,
                                      })
        _res = True
    return _res
  
  
  def setup_embgen_model(self):
    self.maybe_load_pretrained_embgen()
    self._setup_similarity_embeddings()
    return
  

  def setup_pretrained_model(self):
    self._setup_word_embeddings()    
    if self.maybe_load_pretrained():
      self.P("Pretrained model:\n{}".format(
          self.log.GetKerasModelSummary(self.model)))
      self.trained = True
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
  
  def check_labels_set(self, val_labels):
    for obs in val_labels:
      if type(obs) not in [list, tuple, np.ndarray]:
        raise ValueError("LabelSetCheck: All observations must be lists of labels")
      for label in obs:
        if label not in self.dic_labels.keys():
          raise ValueError("LabelSetCheck: Label '{}' not found in valid labels dict".format(label))
    self.P("LabelSetCheck: All {} labels are valid.".format(len(val_labels)))
    return
    
  def initialize(self):
    self.P("Full initialization started ...")
    self._setup_vocabs_and_dicts()
    self._init_hyperparams()
    self.setup_embgen_model()
    self.setup_pretrained_model()
    if self.embeddings is None:
      raise ValueError("Embeddings loading failed!")
    if self.model is None:
      raise ValueError("Model loading failed!")
    if self.embgen_model is None:
      raise ValueError("EmbGen model loading failed!")
    if self.generated_embeddings is None:
      raise ValueError("Generated similarity siamese embeddings loading failed!")
    self.P("Full initialization done.")
    
    
  def tagdict_to_text(self, tags, max_tags=None):
    txt = ''
    cnt = 0
    for k in tags:
      cnt += 1
      txt = txt + "'{}':{:.2f} ".format(k, tags[k])
      if max_tags is not None:
        if cnt >= max_tags:
          break
    return txt

  
  def test_model_on_texts(self, lst_docs, lst_labels, top=5, show=True, DEBUG=False):
    """
    function that calculates (and displays) model validation/testing indicators
    
    inputs:
      lst_docs    : list of documents (each can be a string or a list of strings)
      lst_labels  : list of labels (list) for each document
      top         : max number of tags to generate 
      show        : display stats
    
    returns:
      scalar float with overall accuracy (mean recall)
      
    """
    if not hasattr(self, "train_recall_history"):
      self.train_recall_history = []
    if type(lst_docs) == str:
      lst_docs = [lst_docs]
    if type(lst_labels[0]) == str:
      lst_labels = [lst_labels]
    if len(lst_docs) != len(lst_labels):
      raise ValueError("Number of documents {} must match number of label-groups {}".format(
          len(lst_docs), len(lst_labels)))
    docs_acc = []
    tags_per_doc = []
    if show:
      self.P("")
      self.P("Starting model testing on {} documents".format(len(lst_docs)))
    for idx, doc in enumerate(lst_docs):
      doc_acc = 0
      dct_tags, inputs = self.predict_text(doc, convert_tags=True, 
                                   convert_unknown_words=True, 
                                   top=top,
                                   DEBUG=False,
                                   return_input_processed=True,
                                   verbose=0,
                                   )
      lst_tags = [x.lower() for x in dct_tags]
      gt_tags = lst_labels[idx]
      for y_true in gt_tags:
        if y_true.lower() in lst_tags:
          doc_acc += 1
      if show and DEBUG:
        self.P("  Inputs: ({} chars) '{}...'".format(len(inputs), inputs[:50]))
        self.P("  Predicted: {}".format(self.tagdict_to_text(dct_tags, max_tags=5)))
        self.P("  Labels:    {}".format(gt_tags[:5]))
        self.P("  Match: {}/{}".format(doc_acc, len(gt_tags)))
        self.P("")
      doc_prc = doc_acc / len(gt_tags)      
      tags_per_doc.append(len(gt_tags))
      docs_acc.append(doc_prc)
    overall_acc = np.mean(docs_acc)
    self.train_recall_history.append(round(overall_acc, 2))
    if show:
      self.P("Tagger benchmark on {} documents with {:.1f} avg tags/doc".format(
          len(lst_docs), np.mean(tags_per_doc)))
      self.P("  Overall recall: {:.1f}%".format(overall_acc * 100))
      self.P("  Max doc recall: {:.1f}%".format(np.max(docs_acc) * 100))
      self.P("  Min doc recall: {:.1f}%".format(np.min(docs_acc) * 100))
      self.P("  Med doc recall: {:.1f}%".format(np.median(docs_acc) * 100))
    return overall_acc    

  
if __name__ == '__main__':
  from libraries.logger import Logger
  
  cfg1 = "tagger/brain/configs/config.txt"
  
  use_raw_text = True
  force_batch = True
  use_model_conversion = False
  epochs = 30
  use_loaded = True
  
  l = Logger(lib_name="ALNT",config_file=cfg1)
  l.SupressTFWarn()
  

  
  eng = ALLANTaggerEngine(log=l,)
  
  eng.initialize()
  
    
  l.P("")
  tags, inputs = eng.predict_text("as vrea info despre salarizare daca se poate")
  res = eng.tagdict_to_text(tags)
  l.P("Result: {}".format(res))
  l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))
      

  tags, inputs = eng.predict_text("Aveti cumva sediu si in cluj?")
  res = eng.tagdict_to_text(tags)
  l.P("Result: {}".format(res))
  l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))


  tags, inputs = eng.predict_text("unde aveti birourile in bucuresti?")
  res = eng.tagdict_to_text(tags)
  l.P("Result: {}".format(res))
  l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))


  tags, inputs = eng.predict_text("care este atmosfera de echipa in EY?")
  res = eng.tagdict_to_text(tags)
  l.P("Result: {}".format(res))
  l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))


  tags, inputs = eng.predict_text("in ce zona aveti biroul in Iasi?")
  res = eng.tagdict_to_text(tags)
  l.P("Result: {}".format(res))
  l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))

