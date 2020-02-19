# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:57:46 2020

@author: damia
"""
import numpy as np

from libraries.logger import Logger
from tagger.brain.cv_simple_model_generator import get_model
from word_universe.doc_utils import DocUtils
from functools import partial

from libraries.training import Trainer


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
  
  
  y_train = np.array([dct_label2idx[lbl] for lbl in all_train_labels]).reshape(-1,1)
  y_dev = np.array([dct_label2idx[lbl] for lbl in new_dev_labels]).reshape(-1,1)
  
  
  fct_corpus_to_batch = partial(l.corpus_to_batch,
                                tokenizer_func=tokenizer,
                                dct_word2idx=dct_vocab,
                                max_size=max_size,
                                unk_word_func=None,
                                PAD_ID=tokens_config['<PAD>'],
                                UNK_ID=tokens_config['<UNK>'],
                                left_pad=False,
                                cut_left=False)
  
  
  if USE_EMBEDS:
    X_train = fct_corpus_to_batch(sents=all_train_cv,
                                  get_embeddings=True,
                                  embeddings=np_embeds)
    
    
    X_dev = fct_corpus_to_batch(sents=all_dev_cv,
                              get_embeddings=True,
                              embeddings=np_embeds)
    
    
  else: 
    X_train = fct_corpus_to_batch(sents=all_train_cv,
                                  get_embeddings=False,
                                  embeddings=None)
    
    X_dev = fct_corpus_to_batch(sents=all_dev_cv,
                                get_embeddings=False,
                                embeddings=None)
  
  
  batch_size = 8
  n_batches = X_train.shape[0] // batch_size + 1
  
  nr_train_examples = X_train.shape[0]
  nr_sample_train_examples = int(0.1 * nr_train_examples)
  sample_train_indexes = np.random.choice(np.arange(nr_train_examples),
                                          nr_sample_train_examples,
                                          replace=False)
  X_train_sample, y_train_sample = X_train[sample_train_indexes], y_train[sample_train_indexes]
    
  def train_generator():
    
    while True:
      for step in range(n_batches):
        start = step * batch_size
        end = (step + 1) * batch_size
        
        X = X_train[start:end]
        y = y_train[start:end]
        
        yield X, y
      #endfor
    #endwhile
  #end train_generator
  
  gen = train_generator()

  name = 'CVClf'
  model = get_model(input_shape=(max_size,),
                    n_classes=y_train.max() + 1, 
                    embeddings=np_embeds,
                    name=name,
                    use_gpu=False)

  
  trainer = Trainer(model_name=name,
                    epochs=100,
                    key='dev_acc',
                    key_mode='max',
                    stop_at_fails=30,
                    threshold_progress=0,
                    max_patience=5,
                    max_cooldown=3,
                    lr_decay=0.85,
                    return_history=False,
                    log=l)
  
  trainer.simple_train(model=model,
                       train_gen=gen, steps_per_epoch=n_batches,
                       X_test=X_dev, y_test=y_dev,
                       X_train=X_train_sample, y_train=y_train_sample)
  
  
  



  