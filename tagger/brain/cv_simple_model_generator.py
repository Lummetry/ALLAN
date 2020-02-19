# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:30:58 2020

@author: damia
"""

import tensorflow as tf

def conv1d(tf_x,f,k,s,bn,act,name):
  
  tf_x = tf.keras.layers.Conv1D(filters=f, kernel_size=k, 
                                strides=s, 
                                padding='same' if s==1 else 'valid', 
                                name=name+'_cnv_k{}'.format(k),
                                activation=None,
                                use_bias=not bn)(tf_x)
  if bn:
    tf_x = tf.keras.layers.BatchNormalization(name=name+'_bn')(tf_x)
  tf_x = tf.keras.layers.Activation(act, name=name+'_'+act)(tf_x)
  return tf_x
  

def get_model(input_shape, 
              n_classes,
              embeddings=None, 
              bn=True,
              columns=[(1, 32), (2, 32), (5, 32), (7, 32), (9, 32)],
              act='relu',
              name=''):
  use_embeds = embeddings is not None
  VOCAB_SIZE = embeddings.shape[0]
  EMBED_SIZE = embeddings.shape[1]
  model = None
  
  tf_inp = tf.keras.layers.Input(input_shape, name='inp')
  tf_x = tf_inp
  if use_embeds:    
    _init = tf.keras.initializers.Constant(embeddings)
    tf_x = tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_SIZE, 
                                     embeddings_initializer=_init,
                                     name='embed')(tf_x)
  
  out_columns = []
  level = 1
  tf_column_input = tf_x
  for i,col in enumerate(columns):
    f = col[1]
    k = col[0]
    _name = 'lvl{}_{}'.format(level, i+1)
    tf_x = conv1d(tf_column_input, f=f, k=k, s=1, bn=bn, act=act, name=_name)
    out_columns.append(tf_x)
  
  tf_x = tf.keras.layers.concatenate(out_columns + [tf_column_input], name='concat')
  
  tf_x_gmp = tf.keras.layers.GlobalMaxPooling1D(name='gmp')(tf_x)
  
  # phase 2 conv
  tf_x = conv1d(tf_x, 64, k=3, s=2, bn=bn, act=act, name='lvl2_cnv_k64')  
  tf_x = conv1d(tf_x, 64, k=3, s=2, bn=bn, act=act, name='lvl3_cnv_k64')
  
  tf_x = tf.keras.layers.CuDNNLSTM(128, name='culstm')(tf_x)
  
  tf_x = tf.keras.layers.concatenate([tf_x, tf_x_gmp], name='last_concat')
  
  tf_x = tf.keras.layers.Dense(n_classes, name='readout_lin')(tf_x)
  tf_x = tf.keras.layers.Activation('softmax', 
                                    name='readout_sm')(tf_x)
  tf_out = tf_x
  model = tf.keras.models.Model(tf_inp, tf_out, name=name)
  tf.keras.utils.plot_model(model, 
                            'tagger/brain/test.png',
                            show_shapes=True)
    
  return model  


if __name__ == '__main__':
  import numpy as np
  m = get_model((500,), n_classes=5, 
                embeddings=np.random.rand(1000,128),
                name='test')
  m.summary()

