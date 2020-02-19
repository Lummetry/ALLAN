# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:57:46 2020

@author: damia
"""
import numpy as np
import tensorflow as tf
from functools import partial

from libraries.logger import Logger
from tagger.brain.cv_simple_model_generator import get_model
from word_universe.doc_utils import DocUtils

  

def tokenizer(sentence, dct_vocab, unk_func=None):
  sentence = DocUtils.prepare_for_tokenization(text=sentence,
                                               remove_punctuation=True)
  
  tokens = list(filter(lambda x: x != '', sentence.split(' ')))
  ids = list(map(lambda x: dct_vocab.get(x, unk_func(x)), tokens))
#  tokens = list(map(lambda x: dct_vocab[x], lst_splitted))
  return ids, tokens
  


def check_labels_set(val_labels, dct_labels, exclude=True):
  new_val_labels = []
  for obs in val_labels:
    if type(obs) not in [list, tuple, np.ndarray]:
      raise ValueError("LabelSetCheck: All observations must be lists of labels")
    new_obs = []
    for label in obs:
      if label not in dct_labels.keys():
        _str_info = "LabelSetCheck: Label '{}' not found in valid labels dict".format(label)
        if exclude:
          print(_str_info)
        else:
          raise ValueError(_str_info)
      else:
        new_obs.append(label)
    new_val_labels.append(new_obs)
  return new_val_labels

def load_data(log, training_config):
  train_folder = training_config['FOLDER']
  dev_folder = None
  if 'DEV_FOLDER' in training_config:
    dev_folder = training_config['DEV_FOLDER']
  doc_folder, label_folder = None, None
  doc_ext = training_config['DOCUMENT']
  label_ext = training_config['LABEL']
  if training_config['SUBFOLDERS']['ENABLED']:
    doc_folder = training_config['SUBFOLDERS']['DOCS']
    label_folder = training_config['SUBFOLDERS']['LABELS']

  
  
  train_texts, train_labels = log.load_documents(folder=train_folder,
                                                 doc_folder=doc_folder,
                                                 label_folder=label_folder,
                                                 doc_ext=doc_ext,
                                                 label_ext=label_ext,
                                                 return_labels_list=False,
                                                 exclude_list=['ï»¿'])

  valid_texts, valid_labels = None, None
  if dev_folder is not None:
    valid_texts, valid_labels = log.load_documents(folder=dev_folder,
                                                   doc_folder=doc_folder,
                                                   label_folder=label_folder,
                                                   doc_ext=doc_ext,
                                                   label_ext=label_ext,
                                                   return_labels_list=False,
                                                   exclude_list=['ï»¿'])

  return train_texts, train_labels, valid_texts, valid_labels
  




def get_model(input_shape, embeddings, grams=[1, 2, 5, 7, 9]):
  EMBED_SIZE = embeddings.shape[1]
  model = None
  
  tf_inp = tf.keras.layers.Input(input_shape, 'inp')
  tf_x = tf_inp
  if not USE_EMBEDS:    
    tf_x = tf.keras.layers.Embeddings(EMBED_SIZE, 
                                      embeddings_initializer=tf.keras.initializers.Constant(np_embeds),
                                      name='embed')(tf_x)
  
  columns = []
  level = 1
  for i,gram in enumerate(grams):
    f = 32
    k = gram
    tf_x = tf.keras.layers.Conv1D(filters=f, kernel_size=k, strides=1, padding='same', 
                                  name='c1d_lvl{}_{}'.format(level, i+1))
  return model  



if __name__ == '__main__':
  
  USE_EMBEDS = False
  
  
  l = Logger(lib_name='ACV', config_file='tagger/brain/configs/config_cv_test.txt')
  
  tokens_config = l.config_data['TOKENS']
  training_config = l.config_data['TRAINING']
  tokens = []
  for k,v in tokens_config.items():
    tokens.append(k)
  
  # max observation size
  max_size = 1400

  # train / dev data
  all_train_cv, all_train_labels, all_dev_cv, all_dev_labels = load_data(l, training_config)
  all_train_labels = Logger.flatten_2d_list(all_train_labels)
  assert len(all_train_cv) == len(all_train_labels)
  assert len(all_dev_cv) == len(all_dev_labels)
  n_obs = len(all_train_labels)
  
  # load vocab
  vocab = l.load_pickle_from_models(l.config_data['EMB_MODEL'] + '.index2word.pickle')
  vocab = tokens + vocab
  dct_vocab = {w:i for i,w in enumerate(vocab)}
  
  # load embeddings
  np_embeds = np.load(l.GetModelFile(l.config_data['EMB_MODEL'] + '.wv.vectors.npy'))
  x = np.random.uniform(low=-1,high=1, size=(len(tokens), np_embeds.shape[1]))
  x[tokens_config['<PAD>']] *= 0
  np_embeds = np.concatenate((x,np_embeds),axis=0)
  
  
  # compute labels dict
  dct_label2idx = {lbl:i for i,lbl in enumerate(list(set(all_train_labels)))}
  new_dev_labels = check_labels_set(all_dev_labels, dct_label2idx, exclude=True)
  new_dev_labels = Logger.flatten_2d_list(new_dev_labels)
  
  
  y = np.array([dct_label2idx[lbl] for lbl in all_train_labels]).reshape(-1,1)
  
  if USE_EMBEDS:
    X = l.corpus_to_batch(sents=all_train_cv,
                          tokenizer_func=tokenizer,
                          get_embeddings=True,
                          dct_word2idx=dct_vocab,
                          max_size=max_size,
                          embeddings=np_embeds,
                          unk_word_func=None,
                          PAD_ID=tokens_config['<PAD>'],
                          UNK_ID=tokens_config['<UNK>'],
                          left_pad=False,
                          cut_left=False,
                          )
  else: 
    X = l.corpus_to_batch(sents=all_train_cv,
                          tokenizer_func=tokenizer,
                          get_embeddings=False,
                          dct_word2idx=dct_vocab,
                          max_size=max_size,
                          embeddings=None,
                          unk_word_func=None,
                          PAD_ID=tokens_config['<PAD>'],
                          UNK_ID=tokens_config['<UNK>'],
                          left_pad=False,
                          cut_left=False,
                          )
  
#  input_shape = (max_size,X.shape[-1]) if USE_EMBEDS else (max_size,)
  
#  model = get_model(input_shape, np_embeds)


  