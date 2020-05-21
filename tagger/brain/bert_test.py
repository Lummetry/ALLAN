# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:45:40 2020

@author: Andrei
"""
import numpy as np
import tensorflow as tf

from libraries.logger import Logger

from tagger.brain import utils
from tagger.brain.utils import bert_detokenizer

from transformers import TFBertModel, BertTokenizer


if __name__ == '__main__':
  l = Logger(lib_name='ALBERT', config_file='tagger/brain/configs/config_cv_test.txt')
  GENERATE = False
  
  if GENERATE:
    bert_subfolder = '_allan_data/_ro_bert/20200520'
    bert_folder = l.get_root_subfolder(bert_subfolder)
    assert bert_folder is not None
    
    training_config = l.config_data['TRAINING']
    all_train_cv, all_train_labels, all_dev_cv, all_dev_labels = utils.load_data(l, training_config)
    all_train_labels = Logger.flatten_2d_list(all_train_labels)  
    
    dct_label2idx = {lbl:i for i,lbl in enumerate(list(set(all_train_labels)))}
    new_dev_labels = utils.check_labels_set(all_dev_labels, dct_label2idx, exclude=True)
    new_dev_labels = Logger.flatten_2d_list(new_dev_labels)  
    
    y_train = np.array([dct_label2idx[lbl] for lbl in all_train_labels]).reshape(-1,1)
    y_dev = np.array([dct_label2idx[lbl] for lbl in new_dev_labels]).reshape(-1,1)  
    
    tokenizer = BertTokenizer.from_pretrained(bert_folder)
    l.P("Loading pretrained BERT")
    bert_model = TFBertModel.from_pretrained(bert_folder)
    l.P("Done loading pretrained BERT", show_time=True)
    
    train_data = tokenizer.batch_encode_plus(
      all_train_cv, 
      add_special_tokens=True,
      return_attention_mask=True,
      pad_to_max_length=True,
      max_length=512,
      )
    
    dev_data = tokenizer.batch_encode_plus(
      all_dev_cv, 
      add_special_tokens=True,
      return_attention_mask=True,
      pad_to_max_length=True,
      max_length=512,
      )  
    
    _ids = [tokenizer.encode(x) for x in all_train_cv]
    _lens = [len(x) for x in _ids]
    
    x_train_ids = np.array(train_data['input_ids'])
    x_train_mask = np.array(train_data['attention_mask'])
    x_dev_ids = np.array(dev_data['input_ids'])
    x_dev_mask = np.array(dev_data['attention_mask'])
    
    sample_idx = np.random.randint(0, len(x_train_ids))
    sample_ids = x_train_ids[sample_idx]
    l.P("Ids ({}): {} ...".format(len(sample_ids),sample_ids[:50]))
    sample_tokens = tokenizer.convert_ids_to_tokens(sample_ids)
    sample_text = bert_detokenizer(sample_tokens)
    l.P("Text: \n{}".format(sample_text))
    l.P("Original text:\n{}".format(all_train_cv[sample_idx]))
    
    l.P("Generating train input embeddings...")
    x_train_emb, x_train_clf = bert_model.predict([x_train_ids, x_train_mask])
    l.P("Done generating train input embeddings.", show_time=True)
    l.P("Generating dev input embeddings...")
    x_dev_emb, x_dev_clf = bert_model.predict([x_dev_ids, x_dev_mask])
    l.P("Done generating dev input embeddings.", show_time=True)
  #  xde, xdc = bert_model(x_dev_ids, attention_mask=x_dev_mask)
     
    del bert_model
    
    tf.keras.backend.clear_session()
    
    x_train_input = x_train_emb #[:,0]#x_train_clf
    x_dev_input = x_dev_emb #[:,0]#x_dev_clf
    
    _data = [x_train_input, y_train, x_dev_input, y_dev]
    l.save_pickle_to_data(
        _data,
        'bert_inputs'
        )
  else:
    x_train_input, y_train, x_dev_input, y_dev = l.load_pickle_from_data('bert_inputs')
    
    tf_inp = tf.keras.layers.Input(x_train_input.shape[1:], name='inp')
    tf_x = tf_inp
  #  tf_x = tf.keras.layers.Dense(128, name='d1')(tf_x)
  #  tf_x = tf.keras.layers.Activation('selu')(tf_x)
    grams = [2, 3] #, 5, 7]
    lst_cnv = []
    for gram in grams:
      tf_x_cnv = tf.keras.layers.Conv1D(64, gram)(tf_x)
      tf_x_cnv = tf.keras.layers.Activation('selu')(tf_x_cnv)
      tf_x_cnv = tf.keras.layers.GlobalMaxPooling1D()(tf_x_cnv)
      lst_cnv.append(tf_x_cnv)
    
    tf_x = tf.keras.layers.concatenate(lst_cnv)
    
    tf_x = tf.keras.layers.Dropout(0.5)(tf_x)
    tf_x = tf.keras.layers.Dense(np.unique(y_train).shape[0], activation='softmax')(tf_x)
    tf_out = tf_x
    m = tf.keras.models.Model(tf_inp, tf_out)
    m.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['acc'])
    m.fit(x_train_input, y_train, validation_data=(x_dev_input, y_dev), epochs=100, batch_size=16)
    
    