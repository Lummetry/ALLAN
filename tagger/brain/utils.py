# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:47:34 2020

@author: Andrei
"""
import numpy as np
from word_universe.doc_utils import DocUtils



def bert_detokenizer(word_tokens):
  s = ''
  for w in word_tokens:
    if w == '[PAD]':
      break
    d = ' ' if w[0] != '#' else ''
    s = s + d + w
  return s
    

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


  