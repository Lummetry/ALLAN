# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:30:58 2020

@author: damia
"""

import tensorflow as tf
from libraries.lummetry_layers.gated import GatedDense

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


main_grid = {
    "diremb" : [
        True,
        False
        ],
        
    "bn" : [
        True,
        False
        ],
    
    "cols" : [
        [(1, 64), (2, 64), (5, 128), (7, 128), (9, 128)],
        [(1, 32), (2, 32), (5, 32), (7, 32), (9, 32)],
        [(1, 64), (3, 64), (7, 64)],
        ],
        
    "ph2": [
         3,
         5,
         ],
    
    "fcs" : [
        [(128,True)],
        [(128,True)],
        [],
        ],
        
    "drp" : [
        0.3,
        0.7,
        ],
        
    "pool": [
        "avg",
        "max",
        "both",
        ]
        
        
    
    
    }
  

def get_model(input_shape, 
              n_classes,
              embeddings=None, 
              bn=True,
              columns=[(1, 32), (2, 32), (5, 32), (7, 32), (9, 32)],
              phase2=3,
              fcs=[(128,True)],              
              act='relu',
              pool='both',
              name='',
              drop=0.5,
              use_gpu=True,
              ):
  drop_id = 0
  use_embeds = embeddings is not None
  model = None
  
  if use_gpu:
    LSTM = tf.keras.layers.CuDNNLSTM
  else:
    LSTM = tf.keras.layers.LSTM
  
  tf_inp = tf.keras.layers.Input(input_shape, name='inp')
  tf_x = tf_inp
  if use_embeds:
    VOCAB_SIZE = embeddings.shape[0]
    EMBED_SIZE = embeddings.shape[1]
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
  
  if pool == 'avg':
    tf_x_pool = tf.keras.layers.GlobalAveragePooling1D(name='gap')(tf_x)
  elif pool == 'max':
    tf_x_pool = tf.keras.layers.GlobalMaxPooling1D(name='gmp')(tf_x)
  else:
    tf_x_pool1 = tf.keras.layers.GlobalAveragePooling1D(name='gap')(tf_x)
    tf_x_pool2 = tf.keras.layers.GlobalMaxPooling1D(name='gmp')(tf_x)
    tf_x_pool = tf.keras.layers.concatenate([tf_x_pool1, tf_x_pool2], name='pool_concat')
  
  # phase 2 conv
  for i in range(phase2):
    level += 1
    f = 2**(6+i)
    tf_x = conv1d(tf_x, f=f,  k=3, s=2, bn=bn, act=act, 
                  name='lvl{}_cnv_f{}'.format(level, f))  
 
  tf_x = LSTM(128, name='culstm' if use_gpu else 'lstm')(tf_x)
  
  tf_x = tf.keras.layers.concatenate([tf_x, tf_x_pool], name='last_concat')
  
  if drop > 0:
    drop_id += 1
    tf_x = tf.keras.layers.Dropout(rate=drop, name='drop{}_{}'.format(drop_id, drop).replace('.',''))(tf_x)
  
  for i, lyr in enumerate(fcs):
    units= lyr[0]
    is_gated = lyr[1]
    if is_gated:
      tf_x = GatedDense(units=units, name='gated{}_{}'.format(i+1, units))(tf_x)
    else:
      tf_x = tf.keras.layers.Dense(units, name='lin{}_{}'.format(i+1, units))(tf_x)
      tf_x = tf.keras.layers.Activation(act, name='lin{}_{}'.format(i+1, act))(tf_x)
    if drop > 0:
      drop_id += 1
      tf_x = tf.keras.layers.Dropout(rate=drop, 
                                     name='drop{}_{}'.format(drop_id, drop).replace('.',''))(tf_x)
  
  tf_x = tf.keras.layers.Dense(n_classes, name='readout_lin')(tf_x)
  tf_x = tf.keras.layers.Activation('softmax', 
                                    name='readout_sm')(tf_x)
  tf_out = tf_x
  model = tf.keras.models.Model(tf_inp, tf_out, name=name)
  model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'],
                optimizer='nadam')
  tf.keras.utils.plot_model(model, 
                            'tagger/brain/test.png',
                            show_shapes=True)
    
  return model  


if __name__ == '__main__':
  import numpy as np
  m = get_model((1400,), n_classes=5, 
                embeddings=np.random.rand(1000,128),
                name='test')
  m.summary()

