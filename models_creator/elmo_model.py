import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from collections import Counter
from nltk.tokenize import word_tokenize


class ELMo(object):
    def __init__(self, logger, data_file_name, max_word_length):
        
        self.logger = logger
        self.max_word_length = int(max_word_length)
        self.vocab = Counter()        
        #load training data
        logger.P("Loading text from [{}] ...".format(data_file_name))

        self.raw_text = []
        #load training data
        with open(logger.GetDataFile(data_file_name), encoding="latin-1") as f:
          for line in f:
            self.raw_text.append(line)
        
        del self.raw_text[500:]
        
        logger.P("Dataset of length {} is loaded into memory...".format(len(self.raw_text)))


        #create char2idx and idx2char dictionaries
        CHAR_DICT = 'aăâbcdefghiîjklmnopqrșsțtuvwxyzAĂÂBCDEFGHÎIJKLMNOPQRSȘTȚUVWXYZ0123456789 .!?:,\'%-\(\)/$|&;[]"'
        start_token = '<S>'
        unknown_token = '<UNK>'
        pad_token = '<PAD>'
      
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
        
 
    def corpus_tokenization(self):
      #tokenize input into characters
      self.training_corpus_w = []
      self.training_corpus_c = []
      self.logger.P("Tokenization underway...")
      for sentence in self.raw_text:
        
        char_tokenized_sentence = []
        split_sentence = word_tokenize(sentence)
        
        #update vocabulary
        self.vocab.update(split_sentence)

        for word in split_sentence:
          tokenized_word = [self.char2idx.get('<S>')]
          for char_index in range(self.max_word_length):
            
            if char_index < len(word):
              if word[char_index] not in self.char2idx:
                tokenized_word.append(self.char2idx.get('<UNK>'))
              else:
                tokenized_word.append(self.char2idx.get(word[char_index]))
            else:
              tokenized_word.append(self.char2idx.get('<PAD>'))
            
          char_tokenized_sentence.append(np.array(tokenized_word))
        
        split_sentence.append('<\S>')
        self.training_corpus_w.append(np.array(split_sentence))
        self.training_corpus_c.append(np.array(char_tokenized_sentence))
        
      self.training_corpus_c = np.array(self.training_corpus_c)  
      self.training_corpus_w = np.array(self.training_corpus_w)  
      
      self.logger.P("Tokenized {} sentences, at word and character level(with a max word length of {}) ...".format(len(self.training_corpus_w), self.max_word_length))
      self.logger.P("...Generating a vocabulary size of {}".format(len(self.vocab)))
      
      
      return self.training_corpus_w, self.training_corpus_c
       
    def get_conv_column(self, kernel_size, f_s = 128):
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

    def build_model(self):
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
        
        #bilstm layer 1
        lyr_bidi1 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(512, return_sequences= True), name='bidi_lyr_1')
        tf_elmo_bidi1 = lyr_bidi1(tf_elmo_input)
        
        #bilstm layer 2
        lyr_bidi2 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(512, return_sequences= True), name='bidi_lyr_2')
        tf_elmo_bidi2 = lyr_bidi2(tf_elmo_bidi1)

        #dense layer - size of vocabulary
        lyr_vocab = tf.keras.layers.Dense(units = len(self.vocab), activation = "softmax", name="dense_to_vocab")
        
        tf_embeddings = lyr_vocab(tf_elmo_bidi2)
        model = tf.keras.Model(inputs=tf_input,
                               outputs= tf_embeddings)
        
        model.summary()
        model = model.compile(optimizer = "adam", loss='categorical_crossentropy')
        model = model.fit(x=self.training_corpus_c, y=self.training_corpus_w, batch_size=32, epochs=10, verbose=1)

        return model

