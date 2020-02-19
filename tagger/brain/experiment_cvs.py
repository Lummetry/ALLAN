# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:57:46 2020

@author: damia
"""
import numpy as np

import tensorflow as tf

from libraries.logger import Logger



if __name__ == '__main__':
  
  USE_EMBEDS = False
  
  
  l = Logger(lib_name='ACV', config_file='tagger/brain/configs/config_cs_test.txt')
  
  # max observation size
  max_size = 1400
  
  # load all CVs
  all_cv = None
  
  # load all labels
  all_labels = None
  
  # load embeddings
  np_embeds = None
  EMBED_SIZE = np_embeds.shape[-1]
  
  # load dict
  dct_vocab = None
  
  # tokenizer function (sent, dict, unk_func)
  token_func = None
  
  assert len(all_cv) == len(all_labels)  
  n_obs = len(all_labels)
  
  dct_label2idx = {lbl:i for i,lbl in enumerate(list(set(all_labels)))}
  
  y = np.array([dct_label2idx[lbl] for lbl in all_labels]).reshape(-1,1)
  
  if USE_EMBEDS:
    X = l.corpus_to_batch(sents=all_cv,
                          tokenizer_func=token_func,
                          get_embeddings=True,
                          dct_word2idx=dct_vocab,
                          max_size=max_size,
                          embeddings=np_embeds,
                          unk_word_func=None,
                          PAD_ID=0,
                          UNK_ID=1,
                          left_pad=False,
                          cut_left=False,
                          )
  else: 
    X = l.corpus_to_batch(sents=all_cv,
                          tokenizer_func=token_func,
                          get_embeddings=False,
                          dct_word2idx=dct_vocab,
                          max_size=max_size,
                          embeddings=None,
                          unk_word_func=None,
                          PAD_ID=0,
                          UNK_ID=1,
                          left_pad=False,
                          cut_left=False,
                          )
  
  input_shape = (max_size,X.shape[-1]) if USE_EMBEDS else (max_size,)

  tf_inp = tf.keras.layers.Input(input_shape, 'inp')
  tf_x = tf_inp
  if not USE_EMBEDS:    
    tf_x = tf.keras.layers.Embeddings(EMBED_SIZE, name='embed')(tf_x)