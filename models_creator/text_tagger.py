from models_creator.import_utils import LoadLogger
from models_creator.doc_utils import DocUtils
import tensorflow as tf
import numpy as np
import random
import os

if __name__ == '__main__':
  logger = LoadLogger(lib_name='DOCSUMM', config_file='./config_tagger.txt',
                      use_tf_keras=True)
  
  lstm_units = 256
  fn_emb_matrix = 'demo_20190611/20190611_ep55_vectors.npy'
  emb_matrix = np.load(logger.GetDataFile(fn_emb_matrix))
  
  d = DocUtils(logger, logger.GetDataFile('demo_20190611/20190611_ep55_index2word.pickle'))
  
  d.CreateMasterLabelsVocab(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190226_Production_selection_v0_3/master_labels')
  d.GenerateMasterLabels(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190226_Production_selection_v0_3/master_labels')
  
  X_train, y_train, X_dev, y_dev, ttn, dtn, conv_w = d.GenerateTaggingData(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190226_Production_selection_v0_3/texts')
    
  tf_input = tf.keras.layers.Input(shape=(X_train.shape[1],), name='tf_input') # (batch_size, seq_len)
  EmbLayer = tf.keras.layers.Embedding(input_dim=emb_matrix.shape[0],
                                       output_dim=emb_matrix.shape[1],
                                       weights=[emb_matrix],
                                       trainable=False,
                                       name='emb_layer')
  
  #c1
  lyr_c1_conv1  = tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=2, name='c1_conv1')
  lyr_c1_bn1    = tf.keras.layers.BatchNormalization(name='c1_bn1')
  lyr_c1_act1   = tf.keras.layers.Activation('relu', name='c1_relu1')
  
  lyr_c1_conv2  = tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=2, name='c1_conv2')
  lyr_c1_bn2    = tf.keras.layers.BatchNormalization(name='c1_bn2')
  lyr_c1_act2   = tf.keras.layers.Activation('relu', name='c1_relu2')
  
  lyr_c1_conv3  = tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=2, name='c1_conv3')
  lyr_c1_bn3    = tf.keras.layers.BatchNormalization(name='c1_bn3')
  lyr_c1_act3   = tf.keras.layers.Activation('relu', name='c1_relu3')
  
  lyr_c1_lstm1  = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(units=lstm_units, name='c1_lstm'), name='c1_bidi') 
  
  #c2
  lyr_c2_conv1  = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=3, name='c2_conv1')
  lyr_c2_bn1    = tf.keras.layers.BatchNormalization(name='c2_bn1')
  lyr_c2_act1   = tf.keras.layers.Activation('relu', name='c2_relu1')
  
  lyr_c2_conv2  = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=3, name='c2_conv2')
  lyr_c2_bn2    = tf.keras.layers.BatchNormalization(name='c2_bn2')
  lyr_c2_act2   = tf.keras.layers.Activation('relu', name='c2_relu2')
  
  lyr_c2_lstm1  = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(units=lstm_units, name='c2_lstm'), name='c2_bidi') 
  
  #c3
  lyr_c3_conv1  = tf.keras.layers.Conv1D(filters=128, kernel_size=7, strides=7, name='c3_conv1')
  lyr_c3_bn1    = tf.keras.layers.BatchNormalization(name='c3_bn1')
  lyr_c3_act1   = tf.keras.layers.Activation('relu', name='c3_relu1')

  lyr_c3_lstm1  = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(units=lstm_units, name='c3_lstm'), name='c3_bidi') 

  tf_x = EmbLayer(tf_input) # (batch_size, seq_len, output_dim)

  tf_conv1 = lyr_c1_act3(lyr_c1_bn3(lyr_c1_conv3(lyr_c1_act2(lyr_c1_bn2(lyr_c1_conv2(lyr_c1_act1(lyr_c1_bn1(lyr_c1_conv1(tf_x)))))))))
  tf_conv2 = lyr_c2_act2(lyr_c2_bn2(lyr_c2_conv2(lyr_c2_act1(lyr_c2_bn1(lyr_c2_conv1(tf_x))))))
  tf_conv3 = lyr_c3_act1(lyr_c3_bn1(lyr_c3_conv1(tf_x)))

  tf_x1 = lyr_c1_lstm1(tf_conv1)
  tf_x2 = lyr_c2_lstm1(tf_conv2)
  tf_x3 = lyr_c3_lstm1(tf_conv3)
  
  tf_x = tf.keras.layers.concatenate([tf_x1, tf_x2, tf_x3], name='concatenate')
  
  
  lyr_readout = tf.keras.layers.Dense(units=len(d.dict_master_label2id), 
                                        name='readout', 
  
                                      activation='softmax')
  tf_x = tf.keras.layers.Dropout(rate=0.5, name='dropout')(tf_x)
  tf_x = lyr_readout(tf_x)

  model = tf.keras.Model(inputs=tf_input, outputs=tf_x)
  
  model.summary()
  
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc']) 
    
  model.fit(x=X_train, y=y_train, batch_size=4, epochs=100)
  
  model.save('./models/allan_tagger_1st.h5')

  random_train_texts = list(map(lambda x: (x,1), random.sample(ttn, 5)))
  random_dev_texts   = list(map(lambda x: (x,0), dtn[:5]))
  
  random_texts = random_train_texts + random_dev_texts
  
  for idx,t in enumerate(random_texts):
    inp = np.expand_dims(np.array(conv_w[t[0]]), axis=0)
    logger.P('', noprefix=True)
    is_trained = (t[1] == 1)
    logger.P('Text #{} - trained={}'.format(idx,is_trained), noprefix=True)
    yhat = model.predict(inp)[0]
    best = np.argsort(yhat)[::-1][:3]
    y = d.all_master_labels[t[0]]
    logger.P("Reality: {}".format(list(map(lambda x: d.dict_master_id2label[x], y))), noprefix=True)
    for i in best:
      logger.P('Pred Category: "{}" with probability: {}'.format(d.dict_master_id2label[i], yhat[i]), noprefix=True)
      