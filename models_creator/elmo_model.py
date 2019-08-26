import re
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K


from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report 
from tensorflow.keras.callbacks import Callback


def strip_html_tags(string):
  return re.sub(r'<.*?>', '', string)
  
def string_cleanup(string):
  return re.sub(r'([ \w]*)([!?„”"–,\'\.\(\)\[\]\{\}\:\;\/\\])([ \w]*)', r'\1 \2 \3', string)

def flatten_list(a):
  return [item for sublist in a for item in sublist]

class Metrics(Callback):
  def __init__(self, logger, validation_generator, validation_steps, batch_size, idx2word):
    super().__init__()
    self.logger = logger
    self.validation_generator = validation_generator
    self.validation_steps = validation_steps
    self.batch_size = batch_size
    self.idx2word = idx2word
    
    self.train_preds = []
    self.train_true = []
    self.var_y_true = tf.Variable(0., validate_shape=False)
    self.var_y_pred = tf.Variable(0., validate_shape=False)

  def on_train_begin(self, logs):
    self.val_f1_micros = []
    self.val_f1_macros = []

    self.val_recalls = []
    self.val_precisions = []

  def _get_validation_preds(self):
    val_true = []
    val_pred = []
    
    val_true = np.array(val_true)
    val_pred = np.array(val_pred)
    
    for batch in range(self.validation_steps):
      xVal, yVal = next(self.validation_generator)

      val_true = np.hstack((val_true, np.squeeze(np.asarray(yVal), axis=-1).ravel()))
      val_pred = np.hstack((val_pred, np.argmax(np.asarray(self.model.predict(xVal)).round(), axis=-1).ravel()))

    val_true = np.asarray(val_true, dtype=np.int32)
    val_pred = np.asarray(val_pred, dtype=np.int32)
    
    return val_true, val_pred
    
  def on_epoch_end(self, epoch, logs):
          
    val_true, val_pred = self._get_validation_preds()
    
    target_ids = np.unique(val_true)
    target_names = []
    for i in target_ids:
      target_names.append(self.idx2word[i])
      
    self.logger.P(classification_report(val_true, val_pred, labels=target_ids, target_names=target_names), noprefix=True)
    
    return

class ELMo(object):
    def __init__(self, logger, data_file_name, word2idx_file, max_word_length):
        
        self.logger = logger
        self.word2idx_file = word2idx_file
        self.max_word_length = int(max_word_length)
        self.vocab = Counter()
        
        #load training data
        logger.P("Loading text from [{}] ...".format(data_file_name))
        self.raw_text = []

        with open(self.logger.GetDataFile(data_file_name), encoding="utf-8") as f:
          for line in f:
            line = line.rstrip()
            self.raw_text.append(strip_html_tags(line))
            
        logger.P("Dataset of length {} is loaded into memory...".format(len(self.raw_text)))
        #reduce size for development
        del self.raw_text[2500:]
        
        
        #load word2idx mapping
        logger.P("Loading text from [{}] ...".format(word2idx_file))
        
        start_token = '<S>'
        unknown_token = '<UNK>'
        pad_token = '<PAD>'

        self.word2idx = pd.read_csv(logger.GetDataFile(word2idx_file), header=None)
        #reduce size for development
        self.word2idx = self.word2idx.iloc[:5000]
        
        self.idx2word = self.word2idx.set_index(1).to_dict()[0]
        self.word2idx = dict(zip(self.idx2word.values(), self.idx2word.keys()))
        self.word2idx['<S>']=5001
        self.word2idx['<\S>']= 5002
        self.word2idx['<UNK>']=5003
        self.word2idx[pad_token]=5004
        
        self.idx2word[5001] = '<S>'
        self.idx2word[5002] = '<\S>'
          
        logger.P("{} number of unique words loaded memory...".format(len(self.word2idx)))
        
        #create char2idx and idx2char dictionaries
        CHAR_DICT = 'aăâbcdefghiîjklmnopqrșsțtuvwxyzAĂÂBCDEFGHÎIJKLMNOPQRSȘTȚUVWXYZ0123456789 .!?:,\'%-\(\)/$|&;[]"'
      
        chars = []
        for c in CHAR_DICT:
            chars.append(c)
    
        chars = list(set(chars))
        
        #add special tokens
        chars.insert(0, start_token)
        chars.insert(1, pad_token)
        chars.insert(2, unknown_token)
    
        self.char2idx = dict((c, i) for i, c in enumerate(chars))
        self.idx2char = dict((i, c) for i, c in enumerate(chars))
        
        self.alphabet_size = len(chars)
        
    def word_to_index(self, word):
      return self.word2idx.get(word)
    
    def index_to_word(self, index):
      return self.idx2word.get(index)
    
    # TOKENIZATION FUNCTIONS
    def corpus_tokenization(self):
      #tokenize input into characters
      self.training_corpus_w_str = []
      self.training_corpus_w_idx = []

      self.training_corpus_c = []
      self.logger.P("Tokenization underway...")
      self.logger.P("Processing {} sentences".format(len(self.raw_text)))
     
      start_token_array = np.ones(self.max_word_length)
      start_token_array[0] = self.char2idx.get('<S>')
      for sentence in tqdm(self.raw_text):
        
        char_tokenized_sentence = []
        word_tokenized_sentence = []
        split_sentence = word_tokenize(sentence)
        
        #START TOKEN for each sentence
        char_tokenized_sentence.append(np.array(start_token_array))
        
        #update vocabulary
        self.vocab.update(split_sentence)
        
        for word in split_sentence:
          word_tokenized_sentence.append(self.word_to_index(word))
          char_tokenized_word = []
          for char_index in range(self.max_word_length):
            if char_index < len(word):
              if word[char_index] not in self.char2idx:
                char_tokenized_word.append(self.char2idx.get('<UNK>'))
              else:
                char_tokenized_word.append(self.char2idx.get(word[char_index]))
            else:
              char_tokenized_word.append(self.char2idx.get('<PAD>'))
            
          char_tokenized_sentence.append(np.array(char_tokenized_word))
        
        #append END tokens
        split_sentence.append('<\S>')
        word_tokenized_sentence.append(self.word2idx.get('<\S>'))
        
        self.training_corpus_w_str.append(np.array(split_sentence))
        self.training_corpus_w_idx.append(np.array(word_tokenized_sentence))
        self.training_corpus_c.append(np.array(char_tokenized_sentence))
        
      self.training_corpus_w_str = np.array(self.training_corpus_w_str)
      self.training_corpus_w_idx = np.array(self.training_corpus_w_idx)
      self.training_corpus_c = np.array(self.training_corpus_c) 
      
      self.logger.P("Tokenized {} sentences, at word and character level(with a max word length of {}) ...".format(len(self.training_corpus_w_str), self.max_word_length))
      self.logger.P("...Generating a vocabulary size of {}".format(len(self.vocab)))
      
      assert(len(self.training_corpus_c) == len(self.training_corpus_w_str))
      assert(len(self.training_corpus_w_str) == len(self.training_corpus_w_idx))
      
      self.dict_seq_batches = self.build_doc_length_dict(self.training_corpus_c)
      self.total_seq_lengths = len(list(self.dict_seq_batches.keys()))
      
      return self.training_corpus_w_str, self.training_corpus_c
    
    def create_word2idx_map(self):
      
      self.word2idx = {'<\S>': 0}
      count = 1
      
      for line in tqdm(self.training_corpus_w_str):
        for word in line:
          if word not in self.word2idx.keys():
            self.word2idx[word] = count
            count += 1

      df = pd.DataFrame(self.word2idx.items(), columns=['Word', 'Index'])
      df.to_csv('./rowiki_dialogues_merged_v2_wordindex_df.csv', index=False)


    def token_sanity_check(self, sentence_idx=-1):
      
      if sentence_idx == -1:
        random_item = random.randint(0, len(self.raw_text))
        self.logger.P("Sanity check on random sentence: {}".format(self.raw_text[random_item]))
        tokenized_w = self.training_corpus_w_str[random_item]
        tokenized_c = self.training_corpus_c[random_item]
        
      else:
        tokenized_w = self.training_corpus_w_str[sentence_idx]
        tokenized_c = self.training_corpus_c[sentence_idx]
        
      self.logger.P("Word tokenization: {}".format(tokenized_w))
      
      id_tokenized_w = []
      for word in tokenized_w:
        id_tokenized_w.append(self.word_to_index(word))  
      self.logger.P("Word2Id: {}".format(id_tokenized_w))
      
      back_to_words = ''
      for idx in id_tokenized_w:
        back_to_words = back_to_words + self.index_to_word(idx) + ' '
      
      self.logger.P("Id2Word: {}".format(back_to_words))
      
      
      self.logger.P("Char2Id tokenization: \n{}".format(tokenized_c))
      
      back_to_text = ''
      for word in tokenized_c:
        for char in word:
          back_to_text = back_to_text + self.idx2char.get(char)

        back_to_text = back_to_text + '\n'         
      self.logger.P('Id2Char: {}'.format(back_to_text))
      
    # DATA GENERATOR FUNCTIONS
    
    def build_doc_length_dict(self, doc_list):
      dict_lengths = {}
      for i in range(len(doc_list)):
        try:
          dict_lengths[len(doc_list[i])].append(i)
        except:
          dict_lengths[len(doc_list[i])] = [i]
          
      sorted_dict_lengths = {}
      #sanity check, make sure all documents are here...
      total_docs = 0
      for i in sorted(list(dict_lengths.keys()), reverse=True):
        sorted_dict_lengths[i] = dict_lengths.get(i)
        total_docs += len(dict_lengths.get(i))
      
      del dict_lengths
      
      #ensure that length of initial array is equal to the total of all lengths of the arrays in sorted_dict_lengths
      assert(len(doc_list) == total_docs)
      return sorted_dict_lengths
    
    def build_batch_list(self, batch_size, validation_set_ratio=0.1):
      #iterate through list of indexes of sequences of same lengths
      self.training_batches = []
      for length in list(self.dict_seq_batches.keys()):
        idx_array = self.dict_seq_batches.get(length)
        #generate batches by grouping indexes in sublists of batch_size
        batches = [idx_array[x:x+batch_size] for x in range(0, len(idx_array), batch_size)]
        self.training_batches.append(batches)
        
      self.training_batches = flatten_list(self.training_batches)
      
      split_index = int(len(self.training_batches) * validation_set_ratio)
      #split at index
      self.validation_batches = self.training_batches[split_index:]
      self.number_of_validation_batches = len(self.validation_batches) 
      
      self.training_batches = self.training_batches[:split_index] 
      self.number_of_training_batches = len(self.training_batches)
      
      return self.training_batches
    
    def _format_Xy(self, batch):
      X = []
      y_idx = []
      y_str = []
      for idx in batch:
        #sanity check: assert that the lengths of the rows processed are the same length! 
        assert(len(self.training_corpus_c[idx]) == len(self.training_corpus_w_idx[idx]) == len(self.training_corpus_w_str[idx]))

        X.append(self.training_corpus_c[idx])
        y_str.append(self.training_corpus_w_str[idx])
        y_idx.append(self.training_corpus_w_idx[idx])

      
      X = np.array(X) #shape of X(batch_size, seq_len, alphabet_size)
      y_str = np.array(y_str) #shape of y_str(batch_size, seq_len)
      y_idx = np.array(y_idx) #shape of y_str(batch_size, seq_len)

      return X, y_idx
    
    def train_generator(self):
      while True:
        for batch_idx in range(len(self.training_batches)):
          #turn list of indexes into training data for ELMo
          X, y_idx = self._format_Xy(self.training_batches[batch_idx])
          y_idx = np.expand_dims(y_idx, axis=-1)
          yield X, y_idx
        
    def validation_generator(self):
      while True:
          for batch_idx in range(len(self.validation_batches)):
            #turn list of indexes into training data for ELMo
            X, y_idx = self._format_Xy(self.validation_batches[batch_idx])
            y_idx = np.expand_dims(y_idx, axis=-1)
            yield X, y_idx
            
    # MODEL FUNCTIONS

    def get_conv_column(self, kernel_size, f_s=128):
      #generate convolution column
      nr_collapsed = 1
      nr_convolutions = 0
      last_kernel_size = kernel_size
      while self.max_word_length // nr_collapsed != 1:
          nr_collapsed *= kernel_size
          nr_convolutions += 1
          if nr_collapsed > self.max_word_length:
              nr_collapsed = nr_collapsed // kernel_size
              last_kernel_size = self.max_word_length // nr_collapsed
              break

      tf_inp_model_column = tf.keras.layers.Input(shape=(self.max_word_length, self.alphabet_size), name='inp_model_column_{}'.format(kernel_size))
      tf_x = tf_inp_model_column
      for i in range(nr_convolutions):
          k_s = kernel_size
          if i == nr_convolutions-1:
              k_s = last_kernel_size
          lyr_conv = tf.keras.layers.Conv1D(filters=f_s, kernel_size=k_s, strides=kernel_size, name="c{}_conv{}".format(kernel_size, i+1))
          lyr_bn = tf.keras.layers.BatchNormalization(name="c{}_bn{}".format(kernel_size, i+1))
          lyr_relu = tf.keras.layers.ReLU(name="c{}_relu{}".format(kernel_size,i+1))
          tf_x = lyr_relu(lyr_bn(lyr_conv(tf_x)))
      model = tf.keras.Model(inputs=tf_inp_model_column, outputs=tf_x)
      lyr_td = tf.keras.layers.TimeDistributed(model, name='c{}_td'.format(kernel_size))
      return lyr_td

    def build_charcnn_model(self):
      #character level cnn
      tf_input = tf.keras.layers.Input(shape=(None, self.max_word_length), dtype=tf.int32, name="Input_seq_chars") # (batch_size, seq_len, nr_chars)
      #onehot encoding of characters
      lyr_onehot = tf.keras.layers.Lambda(lambda x: K.one_hot(x, num_classes=self.alphabet_size), name='one_hot_chars')

      tf_onehot_chars = lyr_onehot(tf_input) # (batch_size, seq_len, nr_chars, total_chars)
      
      #convolution columns of different kernel sizes
      lyr_td_c1 = self.get_conv_column(kernel_size = 2)
      lyr_td_c2 = self.get_conv_column(kernel_size = 3)
      lyr_td_c3 = self.get_conv_column(kernel_size = 5)
      lyr_td_c4 = self.get_conv_column(kernel_size = 7)

      tf_c1 = lyr_td_c1(tf_onehot_chars) # (batch_size, seq_len, 1, filters_c1)
      tf_c2 = lyr_td_c2(tf_onehot_chars) # (batch_size, seq_len, 1, filters_c2)
      tf_c3 = lyr_td_c3(tf_onehot_chars) # (batch_size, seq_len, 1, filters_c3)
      tf_c4 = lyr_td_c4(tf_onehot_chars) # (batch_size, seq_len, 1, filters_c4)

      lyr_squeeze = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2), name='squeeze')
      tf_c1_sq = lyr_squeeze(tf_c1) # (batch_size, seq_len, filters_c1)
      tf_c2_sq = lyr_squeeze(tf_c2) # (batch_size, seq_len, filters_c2)
      tf_c3_sq = lyr_squeeze(tf_c3) # (batch_size, seq_len, filters_c3)
      tf_c4_sq = lyr_squeeze(tf_c4) # (batch_size, seq_len, filters_c4)
      
      #all columns concatenated into one - output of character level cnn, input of elmo
      # (batch_size, seq_len, filters_c1 + filters_c2 + filters_c2 + filters_c4)
      tf_elmo_input = tf.keras.layers.concatenate([tf_c1_sq, tf_c2_sq, tf_c3_sq, tf_c4_sq], name='concat_input')
      
      model = tf.keras.Model(inputs=tf_input,
                             outputs= tf_elmo_input)
      
      self.logger.LogKerasModel(model)
      
      return model
   
    def build_elmo_model(self):
      #2 layer bilstm language model
      
      #get input for elmo from char-level cnn
      tf_inputs = tf.keras.layers.Input(shape=(None, self.max_word_length,), dtype='int32', name='char_indices')
      tf_token_representations = self.build_charcnn_model()(tf_inputs)
      
      #bilstm layer 1
      lyr_bidi1 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(512, return_sequences=True), name='bidi_lyr_1')
      
      tf_elmo_bidi1 = lyr_bidi1(tf_token_representations) #(batch_size, seq_len, 512* 2)
      
      #bilstm layer 2
      lyr_bidi2 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(512, return_sequences=True), name='bidi_lyr_2')
      tf_elmo_bidi2 = lyr_bidi2(tf_elmo_bidi1) #(batch_size, seq_len, 512* 2)
      
      #dense layer - size of vocabulary
      lyr_vocab = tf.keras.layers.Dense(units=len(self.word2idx), activation="softmax", name="dense_to_vocab") #(batch_size, seq_len, vocab_size)
      
      tf_readout = lyr_vocab(tf_elmo_bidi2)
      
      model = tf.keras.Model(inputs=tf_inputs,
                             outputs= tf_readout)
      
      self.logger.LogKerasModel(model)

      model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

      return model
      
    def _train(self, batch_size, epochs):
      
      self.logger.P('Start training...')
      self.corpus_tokenization()

      self.token_sanity_check()
      
      self.elmo_model = self.build_elmo_model()
      
      self.build_batch_list(batch_size)
      
      training_steps = self.number_of_training_batches
      validation_steps = self.number_of_validation_batches
      
      valid_metrics = Metrics(self.logger, self.validation_generator(), validation_steps, batch_size, self.idx2word)
      
      self.elmo_model.fit_generator(self.train_generator(), 
                               steps_per_epoch=training_steps, 
                               epochs=epochs,
                               validation_data=self.validation_generator(),
                               validation_steps=validation_steps,
                               callbacks=[valid_metrics])