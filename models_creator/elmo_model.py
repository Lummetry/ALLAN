import re
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine

from models_creator.elmo_custom import Metrics, TimestepDropout

#UTILITIES
def strip_html_tags(string):
  """
  Utility function meant to remove html tags from a given string
  
  Arguments: string
  
  Returns: a string with html tags removed, everything between <> is replaced with a single whitespace
  """
  return re.sub(r'<.*?>', '', string)

def flatten_list(a):
  """
  Utility function that flattens a list of lists
  
  Arguments: a list of lists
  
  Returns: a single list
  """
  return [item for sublist in a for item in sublist]

def perplexity(y_true, y_pred):
  """
  Metric function for model evaluation during training/validation.
  Meant to be used in model.compile metrics parameter
  
  Perplexity is a measure of how well a probability model predicts a sample.
  More information here, on pages 33, 34: https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
  Or just CTRL + F perplexity
  
  Arguments: y_true - ground truth
             y_pred - predictions
  
  Returns: perplexity value
  
  """
  cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
  perplexity = K.pow(2.0, cross_entropy)
  return perplexity
 
class ELMo(object):
  """
  ELMo object.
  
  Meant for: training, with inherent evaluation methods(metrics).
             loading pretrained ELMo weights, generating embeddings as loadable module
             
  """
  def __init__(self, logger, fn_data, fn_word2idx, parameters=None):
    """
    Object Constructor
    
    Arguments: logger,
               fn_data - filename for training corpus
               fn_word2idx - filename for word2idx dictionary
               parameters - None, placeholder for loading parameters from config file
    """
    
    self.logger = logger
    self.fn_data = fn_data
    self.fn_word2idx = fn_word2idx
    self.parameters = parameters
    self.config_data = self.logger.config_data

    self.vocab = Counter()
    
    self._parse_config_data()
    self._load_data()
    self._init_idx_mappings()
  
  def _parse_config_data(self):
    """
    Function for loading parameters from config data.
    
    The parameters are as follows:
    max_word_length: an integer that represents the maximum characters in a word allowed, longer words are discarded
    kernel_size_columns: a list featuring a number of integers that will represent the number of convolution 
                         columns in the character level CNN. Each integer in the list will create a convolution
                         column of the integer's value kernel size.
    charcnn_filter_size: an integer that will set the filter size for each convolutional column
    lstm_hidden_size: the number of hidden layers of the each lstm. Note that this value will be doubled as 
                      the lstms are bidirectional
                      
    Important note regarding these last three values, the length of the kernel size columns list multipled
    by the filter size must be equal to the lstm hidden size. Otherwise the model architecture does not 
    hook up.
    
    dropout_rate: a float in between 0-1, the percentage of dropout applied
    word_dropout_rate: a floate in between 0-1, percentage of timestep dropout applied
    
    clip_value - the interval [-clip_value, clip_value] will be the interval allowed for gradients,
                 larger or smaller values will be clipped to the appropriate clip value, this mitigates
                 the exploding gradient problem
    
    epochs - the number of training epochs ELMo will undergo             
    batch_size - the batch size of sentences ELMo will train with
    
    """
    if self.parameters is None:
      self.parameters = {}
      self.parameters['MAX_WORD_LENGTH'] = self.config_data['MAX_WORD_LENGTH']
      self.parameters['KERNEL_SIZE_COLUMNS'] = self.config_data['KERNEL_SIZE_COLUMNS']
      self.parameters['CHARCNN_FILTER_SIZE'] = self.config_data['CHARCNN_FILTER_SIZE']
      
      self.parameters['LSTM_HIDDEN_SIZE'] = self.config_data['LSTM_HIDDEN_SIZE']
      self.parameters['DROPOUT_RATE'] = self.config_data['DROPOUT_RATE']
      self.parameters['WORD_DROPOUT_RATE'] = self.config_data['WORD_DROPOUT_RATE']    
      self.parameters['CLIP_VALUE'] = self.config_data['CLIP_VALUE']
      
      self.parameters['EPOCHS'] = self.config_data['EPOCHS']
      self.parameters['BATCH_SIZE'] = self.config_data['BATCH_SIZE']
  
    return

  def _load_data(self):
    """
    Function that loads the training data from the fn_data file.
    The raw text is stored in a list of lists where each list is a line.
    Each line is stripped of \n and html tags.
    
    For development purposes, a line trimming the dataset is kept in,
    remove it when training on the whole dataset.
    """
    #load training data
    self.logger.P("Loading text from [{}] ...".format(self.fn_data))
    self.raw_text = []

    with open(self.logger.GetDataFile(self.fn_data), encoding="utf-8") as f:
      for line in f:
        line = line.rstrip()
        self.raw_text.append(strip_html_tags(line))
        
    #reduce size for development
    del self.raw_text[2500:]

    self.logger.P("Dataset of length {} is loaded into memory...".format(len(self.raw_text)))

    
  def _init_idx_mappings(self):
    """
    Function that initializes the word2idx and char2idx mappings.
    
    Creates chard2idx and idx2char dictionaries.
    
    Loads the word2idx dictionary from the fn_word2idx file 
    and creates a idx2word dictionary as well.
    """
    #load word2idx mapping
    self.logger.P("Loading text from [{}] ...".format(self.fn_word2idx))
    self.word2idx = pd.read_csv(self.logger.GetDataFile(self.fn_word2idx), header=None)
    
    #reduce size for development
    self.word2idx = self.word2idx.iloc[:10000]
    
    self.idx2word = self.word2idx.set_index(1).to_dict()[0]
    del self.idx2word['Index']
    self.idx2word = {int(k):v for k,v in self.idx2word.items()}
    self.word2idx = dict(zip(self.idx2word.values(), self.idx2word.keys()))
      
    self.logger.P("{} number of unique words loaded memory...".format(len(self.word2idx)))
    
    #create char2idx and idx2char dictionaries
    CHAR_DICT = 'aăâbcdefghiîjklmnopqrșsțtuvwxyzAĂÂBCDEFGHÎIJKLMNOPQRSȘTȚUVWXYZ0123456789 .!?:,\'%-\(\)/$|&;[]"'
  
    chars = []
    for c in CHAR_DICT:
        chars.append(c)

    chars = list(set(chars))
    start_token = '<S>'
    pad_token = '<PAD>'
    unknown_token = '<UNK>'
    
    #add special tokens
    chars.insert(0, start_token)
    chars.insert(1, pad_token)
    chars.insert(2, unknown_token)

    self.char2idx = dict((c, i) for i, c in enumerate(chars))
    self.idx2char = dict((i, c) for i, c in enumerate(chars))
    self.alphabet_size = len(chars)

  def word_to_index(self, word):
    """
    Function that returns the index of a word in the word2idx mapping.
    If the word is not in the mapping, the index of the unknown token is returned
    """
    if word not in self.word2idx.keys():
      return self.word2idx.get('<UNK>')
    else:
      return self.word2idx.get(word)
  
  def index_to_word(self, index):
    """
    Function that returns the word of an index in the idx2word mapping.
    If the index is not in the mapping, the unknown token is returned
    """
    if index not in self.idx2word.keys():
      return self.idx2word.get(3)
    else:
      return self.idx2word.get(index)
  
  # TOKENIZATION FUNCTIONS
  def atomic_tokenization(self, sentence):
    """
    Atomic tokenization function
    
    IMPORTANT NOTE:
      Language model objective means predicting the next word of the sentence.
      The data is constructed such that:
        Character tokenization does not have start token and ends with an end token
        word tokenization starts with start token and has no end token.
      
      This means that during training the data does not need to be reconstructed for language model 
      objective. The character level tokenization is the input(X) and the word level tokenization 
      is the ground truth(y).
      
      e.g. Am plecat
          character tokenization:
          [start_token pad pad pad pad pad pad pad]
          [1 2 pad pad pad pad pad pad]
          [7 8 9 10 1 4 pad pad]
          
          word tokenization:
          [15 23 end_token]
          
    In this example, we can see that for the first word in the input - the start token, the first 
    ground truth is "am", for the second word in the input "am" the ground truth is "plecat" and 
    for the last word "plecat" the ground truth is the end token.
          
    Arguments: a string, a sentence, of any length of words
    
    Returns: char_tokenization: a list of lists where each list represents the word and each item in the 
                                list is a the character index of each character in the word.
                                These are padded until the max_word_length.
             
             word_idx_tokenization: a list containing the indexes of each word in the word2idx mapping
             
             split_sentence: a list containing the string tokens used in the tokenizations
  
    """
    char_tokenized_sentence = []
    word_tokenized_sentence = []
    split_sentence = word_tokenize(sentence)
    
    #first line of every sentence tokeinzed for chars
    start_token_array = np.ones(self.parameters['MAX_WORD_LENGTH'])
    start_token_array[0] = self.char2idx.get('<S>')
    
    #start token for each sentence
    char_tokenized_sentence.append(np.array(start_token_array))
    
    #update vocabulary
    self.vocab.update(split_sentence)

    for word in split_sentence:
      word_tokenized_sentence.append(self.word_to_index(word))
      char_tokenized_word = []
      for char_index in range(self.parameters['MAX_WORD_LENGTH']):
        if char_index < len(word):
          if word[char_index] not in self.char2idx:
            char_tokenized_word.append(self.char2idx.get('<UNK>'))
          else:
            char_tokenized_word.append(self.char2idx.get(word[char_index]))
        else:
          char_tokenized_word.append(self.char2idx.get('<PAD>'))

      char_tokenized_sentence.append(np.array(char_tokenized_word))

    #append END tokens
    split_sentence.append('<EOS>')
    word_tokenized_sentence.append(self.word2idx.get('<EOS>'))

    return char_tokenized_sentence, word_tokenized_sentence, split_sentence
  
  def corpus_tokenization(self):
    """
    Tokenization on entire dataset.
    Applies the atomic_tokenization function to each line in the training corpus.
    
    The outputs of the atomic_tokenization are appended to lists of the ELMo object:
      
      training_corpus_w_str - a np array of np arrays where each array contains the tokenized sentence
                              as a string 
      training_corpus_w_str - a np array of np arrays where each array contains the tokenized sentence
                              as wordidx
      training_corpus_c - a np array of np array where each array contains the character tokenized 
                          sentence                        
    """
    #tokenize input into characters
    self.training_corpus_w_str = []
    self.training_corpus_w_idx = []
    
    self.training_corpus_c = []
    self.logger.P("Tokenization underway...")
    self.logger.P("Processing {} sentences".format(len(self.raw_text)))
   
    for sentence in tqdm(self.raw_text):
      char_tokenized_sentence, word_tokenized_sentence, split_sentence = self.atomic_tokenization(sentence)
      
      self.training_corpus_w_str.append(np.array(split_sentence))
      self.training_corpus_w_idx.append(np.array(word_tokenized_sentence))
      self.training_corpus_c.append(np.array(char_tokenized_sentence))
      
    self.training_corpus_w_str = np.array(self.training_corpus_w_str)
    self.training_corpus_w_idx = np.array(self.training_corpus_w_idx)
    self.training_corpus_c = np.array(self.training_corpus_c) 
    
    self.logger.P("Tokenized {} sentences, at word and character level(with a max word length of {}) ...".format(len(self.training_corpus_w_str), self.parameters['MAX_WORD_LENGTH']))
    self.logger.P("...Generating a vocabulary size of {}".format(len(self.vocab)))
    
    assert(len(self.training_corpus_c) == len(self.training_corpus_w_str))
    assert(len(self.training_corpus_w_str) == len(self.training_corpus_w_idx))
    
    self.dict_seq_batches = self.build_doc_length_dict(self.training_corpus_c)
    self.total_seq_lengths = len(list(self.dict_seq_batches.keys()))
    
    return self.training_corpus_w_str, self.training_corpus_c
  
  def create_word2idx_map(self):
    """
    Create word to index mapping from data corpus.
    This function is meant to be used before a training session.
    One must generate the word2idx map over the entire corpus, this will yield a idx mapping 
    which will then be loaded as a parameter to the ELMo object.
    
    """
    self.word2idx = {'<S>': 0,
                     '<EOS>': 1,
                     '<PAD>': 2,
                     '<UNK>': 3}
    count = 4
    for line in tqdm(self.raw_text):
      for word in word_tokenize(line):
        if word not in self.word2idx.keys():
          self.word2idx[word] = count
          count += 1

    df = pd.DataFrame(self.word2idx.items(), columns=['Word', 'Index'])
    df.to_csv('./rowiki_dialogues_merged_v2_wordindex_df.csv', index=False)

  def word_length_distrib(self):
    """
    Displays the distribution of word lengths in word2idx file.
    """
    #create list of word lengths from keys of word2idx dictionary
    w_lens = list(self.word2idx.keys())
    w_lengths = []
    for i in w_lens:
      if type(i) is str:
        w_lengths.append(float(len(i)))
    
    #option to supress scientific notation in pandas
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
    #create dataframe with wordlengths
    df_word_len = pd.DataFrame(columns=['len'])
    df_word_len.len = w_lengths
    df_words = df_word_len.astype('float32')
    
    #print word length distribution
    self.logger.P("The distribution of word lengths is:\n {}".format(df_words.describe()))
    
  def token_sanity_check(self, sentence_idx=None):
    """ 
    Sanity check for word-level and char-level tokenization.
    
    Arguments: sentence_idx: the index of a list in the training_corpus_w_str
                            if the sentence_idx=None, the sentence is randomly chosen.

    Displays the word tokenized sentence, both string and idx, and character tokenized sentence.
    """
    if sentence_idx is None:
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
    """
    Builds a dictionary where every key is a document length, and every value is a list of document indexes
    of that value.
    
    Arguments: doc_list: a list of lists where each list is a document(build with training_corpus_w_str in mind)
    
    Returns: a dictionary with keys in ascending order, every key is a length, every value is a list of document indexes
    of that length.
    """
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
    
    sentence_lengths = list(dict_lengths.keys())
    df_sentence_distrib = pd.DataFrame(columns=["len"])
    df_sentence_distrib.len = sentence_lengths
    self.logger.P("The distribution of sentence lengths is: \n {}".format(df_sentence_distrib.describe()))
    
    #remove unneccessary details from memory
    del dict_lengths
    del sentence_lengths
    del df_sentence_distrib
    
    #ensure that length of initial array is equal to the total of all lengths of the arrays in sorted_dict_lengths
    assert(len(doc_list) == total_docs)
    return sorted_dict_lengths
  
  def build_batch_list(self, batch_size, validation_set_ratio=0.1):
    """
    Builds a list of training batches.
    
    The training batches list contains a list of lists where each list is populated by document indexes.
    During training, this list will be used to pull documents from the corpus based on their index.
    
    For every item in the dictionary of document lengths, a batch is constructed 
    by adding batch_size documents of the same length. If there are not enough items to construct a 
    batch, the batch will be processed as such.
    
    The function also generates a batcified validation set.
    
    Arguments: batch_size
               validation_set_ratio: the percentage of the corpus used for validation
               
    Returns: self.training_batches- a list of lists where each list is populated by document indexes, all
                                    documents of a list share the same length
    """
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
    """
    Function that brings a batch into the required format for training.
    
    To understand how the data is arranged, see the documenation for 
    the atomic tokenization function.
    
    Arguments: batch - a list of indexes to sentences(documents in the corpus)
    
    Returns X: character tokenized version of each sentence/document in the batch
            y_idx: word tokenized version of each sentence in the batch
            
    """
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
    """
    Training generator yielding batches from self.training_batches
    """
    while True:
      for batch_idx in range(len(self.training_batches)):
        #turn list of indexes into training data for ELMo
        X, y_idx = self._format_Xy(self.training_batches[batch_idx])
        y_idx = np.expand_dims(y_idx, axis=-1)
        yield X, y_idx
      
  def validation_generator(self):
    """
    Validation generator yielding batches from self.validation_batches
    """
    while True:
        for batch_idx in range(len(self.validation_batches)):
          #turn list of indexes into training data for ELMo
          X, y_idx = self._format_Xy(self.validation_batches[batch_idx])
          y_idx = np.expand_dims(y_idx, axis=-1)
          yield X, y_idx
          
  # MODEL FUNCTIONS
  def get_conv_column(self, kernel_size, f_s=128):
    """
    Generate a single convolution column
    Convolutional column takes one-hot representation of characters and reduces the dimensionality
    with kernel_size until (batch_size, seq_len, 1, alphabet_size)
    """
    nr_collapsed = 1
    nr_convolutions = 0
    last_kernel_size = kernel_size
    while self.parameters['MAX_WORD_LENGTH'] // nr_collapsed != 1:
        nr_collapsed *= kernel_size
        nr_convolutions += 1
        if nr_collapsed > self.parameters['MAX_WORD_LENGTH']:
            nr_collapsed = nr_collapsed // kernel_size
            last_kernel_size = self.parameters['MAX_WORD_LENGTH'] // nr_collapsed
            break

    tf_inp_model_column = tf.keras.layers.Input(shape=(self.parameters['MAX_WORD_LENGTH'], self.alphabet_size), name='inp_model_column_{}'.format(kernel_size))
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

  def build_charcnn_model(self, kernel_sizes):
    """ 
    Character Level CNN
    Builds a character level CNN, takes a list of kernel sizes that define the architecture of the CNN, 
    where each kernel size translates into a conv_column of kernel size i.
    
    The output of the char level cnn is the length of the kernel size list multipled by the filter size,
    it must match the size of the hidden layers of the lstms.
    """
    #input size (batch_size, seq_len, nr_chars)
    tf_input = tf.keras.layers.Input(shape=(None, self.parameters['MAX_WORD_LENGTH']), 
                                     dtype=tf.int32, 
                                     name="Input_seq_chars") 
    
    #onehot encoding of characters: (batch_size, seq_len, nr_chars, total_chars)
    lyr_onehot = tf.keras.layers.Lambda(lambda x: K.one_hot(x, num_classes=self.alphabet_size), 
                                        name='one_hot_chars')

    tf_onehot_chars = lyr_onehot(tf_input) 
    
    #convolution columns of different kernel sizes: (batch_size, seq_len, 1, filters_c[i])
    lyr_td_c = []
    for i in kernel_sizes:
      lyr_td_c.append(self.get_conv_column(i, self.parameters['CHARCNN_FILTER_SIZE'])) 

    tf_c = []
    for i in range(len(kernel_sizes)):
      tf_c.append(lyr_td_c[i](tf_onehot_chars))
    
    #reducing the dimensionality at the axis with 1: (batch_size, seq_len, filters_c[i])
    lyr_squeeze = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2), 
                                         name='squeeze')
    tf_c_sq = []
    for i in tf_c:
      tf_c_sq.append(lyr_squeeze(i)) 
    
    #all columns concatenated into one - output of character level cnn, input of elmo
    # (batch_size, seq_len, filters_c1 + filters_c2 + filters_c2 + filters_c4)
    tf_elmo_input = tf.keras.layers.concatenate(tf_c_sq, 
                                                name='concat_input')
    
    model = tf.keras.Model(inputs=tf_input,
                           outputs= tf_elmo_input,
                           name = 'char_cnn')
    
    self.logger.LogKerasModel(model)
    
    return model
 
  def build_elmo_model(self):
    """
    Builds the ELMo architecture.
    
    Input is the charlevel cnn output, and which then goes into two bidirectional lstms and finally
    to a dense layer for next word prediction.
    
    Model uses adam for optimization, sparse categorical crossentropy as loss function, and 
    sparse categorical accuracy and perplexity as metrics.
    
    This is a multilabel classification problem so recall and accuracy cannot be computed for all classes,
    rather for each individual class. Since this is computationally expensive, these metrics will be 
    displayed in during validation for the validation set.
    """
    #2 layer bilstm language model
    
    #get input for elmo from char-level cnn
    tf_inputs = tf.keras.layers.Input(shape=(None, self.parameters['MAX_WORD_LENGTH'],), 
                                      dtype='int32', 
                                      name='char_indices')
    
    tf_token_representations = self.build_charcnn_model(self.parameters['KERNEL_SIZE_COLUMNS'])(tf_inputs)
    
    #dropout layers
    tf_token_dropout = tf.keras.layers.SpatialDropout1D(self.parameters['DROPOUT_RATE'], name='spatial_dropout')(tf_token_representations)
    tf_token_dropout = TimestepDropout(self.parameters['WORD_DROPOUT_RATE'], name='timestep_dropout')(tf_token_dropout)
    
    #bilstm layer 1 of shape (batch_size, seq_len, lstm hidden units* 2)
    lyr_bidi1 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(self.parameters['LSTM_HIDDEN_SIZE'],
                                                                        recurrent_constraint=tf.keras.constraints.MinMaxNorm(-1*self.parameters['CLIP_VALUE'],
                                                                                                                             self.parameters['CLIP_VALUE']),
                                                                        kernel_constraint=tf.keras.constraints.MinMaxNorm(-1*self.parameters['CLIP_VALUE'],
                                                                                                                             self.parameters['CLIP_VALUE']),
                                                                        return_sequences=True),
                                                                        name='bidi_lyr_1')
    tf_elmo_bidi1 = lyr_bidi1(tf_token_dropout)
    
    #bilstm layer 2 of shape (batch_size, seq_len, lstm hidden units* 2)
    lyr_bidi2 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(self.parameters['LSTM_HIDDEN_SIZE'],
                                                                        recurrent_constraint=tf.keras.constraints.MinMaxNorm(-1*self.parameters['CLIP_VALUE'],
                                                                                                                             self.parameters['CLIP_VALUE']),
                                                                        kernel_constraint=tf.keras.constraints.MinMaxNorm(-1*self.parameters['CLIP_VALUE'],
                                                                                                                             self.parameters['CLIP_VALUE']),
                                                                        return_sequences=True), 
                                                                        name='bidi_lyr_2')
    tf_elmo_bidi2 = lyr_bidi2(tf_elmo_bidi1)
    
    #dense layer of shape (batch_size, seq_len, vocab_size)
    lyr_vocab = tf.keras.layers.Dense(units=len(self.word2idx), 
                                      activation="softmax", 
                                      name="dense_to_vocab") 
    tf_readout = lyr_vocab(tf_elmo_bidi2)
    
    model = tf.keras.Model(inputs=tf_inputs,
                           outputs= tf_readout)
    self.logger.LogKerasModel(model)
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['sparse_categorical_accuracy',perplexity])
    return model
  
  def train(self):
    """
    Function that performs the training of the network.
    
    It applies fit_generator on the elmo_model by using the train_generator function described above.
    """
    epochs = self.parameters['EPOCHS']
    batch_size = self.parameters['BATCH_SIZE']
    
    self.logger.P('Start training...')
    self.corpus_tokenization()

    self.token_sanity_check()
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
                                  validation_steps=validation_steps)
  
  def save_elmo_weights(self, filename):
    """
    Method for saving elmo weights to pickle file.
    
    Arguments: filename: name of the file under whch the weights are saved
    """
    #model for saving elmo weights to pkl
    elmo_layers = [layer.name for layer in self.elmo_model.layers]
    self.logger.P('Saving the model weights...')
    self.logger.SaveKerasModelWeights(filename, self.elmo_model, elmo_layers)
  
  def load_pretrained_elmo(self, filename):
    """
    Method for loading pretrained elmo weights from pickle file
    """
    #load pretrained model from saved model weights
    self.elmo_model = self.build_elmo_model()
    elmo_layers = [layer.name for layer in self.elmo_model.layers]
    self.logger.P('Loading ELMo from weights file...')
    self.logger.LoadKerasModelWeights(filename, self.elmo_model, elmo_layers)
    
  def get_elmo(self, sentence):
    """
    The core functionality of ELMo.
    Once the model is trained, one can predict on a single input sentence 
    and the outputs of the hidden layers will yield the ELMo embeddings that
    are usable further for any purpose.
    
    Arguments: sentence: a string 
  
    Returns: a list with the outputs of each ELMo layer:
             layer0: Char_CNN
             layer1: LSTM1
             layer2: LSTM2
    Returns ELMo embeddings for a given sentence.
    """
    char_tokenzized_sentence, _, _ = self.atomic_tokenization(sentence)
    X = np.expand_dims(char_tokenzized_sentence, axis=0)
    self.logger.P('Generating ELMo embeddings for sentence: {}'.format(sentence))
    
    #get elmo layer 0
    get_charcnn_output = K.function([self.elmo_model.layers[0].input],
                                    [self.elmo_model.layers[-4].output])
    
    layer0_output = get_charcnn_output([X])[0]
    layer0_output = np.asarray(layer0_output)
    layer0_output = layer0_output[0][1:]
    self.logger.P('shape of layer {}'.format(layer0_output.shape))
    self.logger.P('layer 2:\n {}'.format(layer0_output))
    
    #get elmo layer 1
    get_bilstm1_output = K.function([self.elmo_model.layers[0].input],
                                    [self.elmo_model.layers[-3].output])
    
    layer1_output = get_bilstm1_output([X])[0]
    layer1_output = np.asarray(layer1_output)
    layer1_output = layer1_output[0][1:]
    self.logger.P('shape of layer {}'.format(layer1_output.shape))
    self.logger.P('layer 2:\n {}'.format(layer1_output))
    
    #get elmo layer 2 
    get_bilstm2_output = K.function([self.elmo_model.layers[0].input],
                                    [self.elmo_model.layers[-2].output])
    
    layer2_output = get_bilstm2_output([X])[0]
    layer2_output = np.asarray(layer2_output)
    layer2_output = layer2_output[0][1:]
    self.logger.P('shape of layer {}'.format(layer2_output.shape))
    self.logger.P('layer 2:\n {}'.format(layer2_output))
    
    return [layer0_output, layer1_output, layer2_output]
  
  def heat_map_sentence_similarity(self, sentence):
    """
    Function for visualization of sentence word similarity.
    It take a single string sentence, computes ELMo representations 
    and displays a matrix where each cell is the cosine distance between the
    two words(x value and y value). The darker colours represent higher similarity.
    """
    _, _, tokens = self.atomic_tokenization(sentence)
    _, layer_1, layer_2 = self.get_elmo(sentence)
    #fill similarity matrix
    data = np.empty(shape=(layer_1.shape[0],layer_1.shape[0]))
    for x in range(layer_1.shape[0]):
      for y in range(layer_1.shape[0]):
        data[x][y] = cosine(layer_1[x], layer_1[y])

    heat_map = sns.heatmap(data,
                           xticklabels=tokens[:-1],
                           yticklabels=tokens[:-1])
    plt.show()
