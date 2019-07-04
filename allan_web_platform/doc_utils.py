import os
import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

class DocUtils():
  def __init__(self, gensim_i2v, max_nr_words=None, max_nr_chars=None):

    with open(gensim_i2v, 'rb') as handle:
      id2word = pickle.load(handle)

    self.dict_id2word = {}
    self.dict_word2id = {}
    for k in range(len(id2word)):
      self.dict_id2word[k] = id2word[k]

    last_index = len(self.dict_id2word) - 1
    self.dict_id2word[last_index + 1] = '<PAD>'
    self.dict_id2word[last_index + 2] = '<START>'
    self.dict_id2word[last_index + 3] = '<END>'
    self.dict_word2id = {v:k for k,v in self.dict_id2word.items()}

    full_voc = "".join([chr(0)] + [chr(i) for i in range(32, 127)] + [chr(i) for i in range(162,256)])
    self.dict_char2id = {full_voc[i]:i for i in range(len(full_voc))}
    self.dict_id2char = {v:k for k,v in self.dict_char2id.items()}
    
    self.dict_label2id = {}
    self.dict_id2label = {}
    
    self.all_labels = {}
    
    self.end_char_id = self.dict_word2id['<END>']
    self.start_char_id = self.dict_word2id['<START>']
    
    self.max_nr_words = max_nr_words
    self.max_nr_chars = max_nr_chars
    return

  def prepare_for_tokenization(self, string):
    return re.sub(r'([ \w]*)([!?„”"–,\'\.\(\)\[\]\{\}\:\;\/\\])([ \w]*)', r'\1 \2 \3', string)


  def strip_html_tags(self, string):
    return re.sub(r'<.*?>', '', string)


  def CreateLabelsVocab(self, fn):
    with open(fn, 'rt') as handle:
      labels = handle.read().splitlines()

    self.dict_label2id = {labels[i]: i for i in range(len(labels))}
    self.dict_id2label = {v:k for k,v in self.dict_label2id.items()}
    return

  
  def GenerateLabels(self, path):
    assert self.dict_label2id != {}
    self.all_labels = {}
    for file in os.listdir(path):
      full_path = os.path.join(path, file)
      with open(full_path, 'rt') as handle:
        labels = handle.read().splitlines()
        labels = list(map(lambda x: self.dict_label2id[x], labels))

      
      self.all_labels[file] = labels
    return

  def GenerateBatches(self, path, use_characters=True, use_labels=True, eps_words=10, eps_characters=30):
    conversations_w, conversations_c = self.tokenize_conversations(path=path,
                                                                   use_characters=use_characters,
                                                                   eps_words=eps_words,
                                                                   eps_characters=eps_characters)
    
    batches = []

    if use_characters: assert len(conversations_w) == len(conversations_c)
    if use_labels: assert self.all_labels != {}

    for idx_conv in conversations_w:
      current_conversation_w = conversations_w[idx_conv]
      if use_characters:
        current_conversation_c = conversations_c[idx_conv]
        assert len(current_conversation_w) == len(current_conversation_c)
      
      if use_labels:
        current_labels = self.all_labels[idx_conv]
        assert len(current_labels) == len(current_conversation_w), "Conv '{}'".format(idx_conv)

      for num in range(2, len(current_conversation_w)):
        if num % 2 != 0: continue
        target = [self.dict_word2id['<START>']] + current_conversation_w[num] + [self.dict_word2id['<END>']]
        target = np.array(target)
        
        crt_corpus_w = current_conversation_w[:num]
        if use_characters: crt_corpus_c = current_conversation_c[:num]

        new_corpus = []
        for i,c in enumerate(crt_corpus_w):
          d = c[:self.max_nr_words]
          d = d + [self.dict_word2id['<PAD>']] * (self.max_nr_words - len(d))
  
          if use_characters:
            e = crt_corpus_c[i]
            f = e[:self.max_nr_chars]
            f = f + [self.dict_char2id['\0']] * (self.max_nr_chars - len(f))
            d = d + f

          new_corpus.append(d)
        new_corpus = np.array(new_corpus)
        
        if not use_labels:
          batches.append((new_corpus, target))
        else:
          batches.append((new_corpus, target, current_labels[:num]))

    return batches


  def tokenize_conversations(self, path, use_characters=True, eps_words=10, eps_characters=30):
    num_chars_distribution = []
    num_words_distribution = []
    conversations_w = {}
    conversations_c = {}
    self.unknown_words = {}

    for file in os.listdir(path):
      full_path = os.path.join(path, file)
      current_conversation_w = []
      current_conversation_c = []
      with open(full_path, "r") as f:
        for line in f.readlines():
          if line == '\n': continue
          if line[-1] == '\n': line = line[:-1]
          characters = []
          characters.extend(line)
          num_chars_distribution.append(len(characters))

          line_mod = self.prepare_for_tokenization(line)
          line_mod = line_mod.split()
          num_words_distribution.append(len(line_mod))
          tokens_w = []
          for t in line_mod:
            try:
              tokens_w.append(self.dict_word2id[t])
            except:
              tokens_w.append(self.dict_word2id['<UNK>'])
              if t not in self.unknown_words:
                self.unknown_words[t] = 1
              else:
                self.unknown_words[t] += 1
          #endfor

          tokens_c = []
          for c in characters:
            tokens_c.append(self.dict_char2id[c])
          
          current_conversation_c.append(tokens_c)
          current_conversation_w.append(tokens_w)
        #endfor
      #endwith
      conversations_c[file] = current_conversation_c
      conversations_w[file] = current_conversation_w
    #endfor

    num_chars_distribution = np.array(num_chars_distribution).reshape(-1,1)
    num_words_distribution = np.array(num_words_distribution).reshape(-1,1)

    df_distrib = pd.DataFrame(data=np.concatenate((num_words_distribution, num_chars_distribution), axis=1),
                              columns=['Words', 'Characters']).describe()

    print("Distributions descriptions:\n" + df_distrib.to_string())

    self.max_nr_words = int(df_distrib.loc['max']['Words']) + eps_words
    self.max_nr_chars = int(df_distrib.loc['max']['Characters']) + eps_characters

    print("Setting max nr. words to {}".format(self.max_nr_words))
    print("Setting max nr. chars to {}".format(self.max_nr_chars))
    
    if use_characters:
      return conversations_w, conversations_c
    else:
      return conversations_w, None
    
    
  def input_word_text_to_tokens(self, lines, use_characters=True):        
    tokens = []
    if type(lines) is not list:
      lines  = lines.splitlines()

    for line in lines:
      idText = []
      words = word_tokenize(line)
      
      words = list(filter(lambda x: x != '<', words))
      words = list(filter(lambda x: x != '>', words))
      if 'UNK' in words:
        idx = words.index('UNK')
        words[idx] = '<UNK>'

      for idx, item in enumerate(words):
        if item == 'UNK':
          words[idx] = '<UNK>'
        if item == 'NAME':
          words[idx] = '<NAME>'
        if item == 'NAMEBOT':
          words[idx] = '<NAMEBOT>'

      for word in words:
        try:
          idText.append(self.dict_word2id[word])
        except:
          idText.append(self.dict_word2id['<UNK>'])

      idText_pad = [self.dict_word2id['<PAD>']] * (self.max_nr_words - len(idText))
      idText = idText + idText_pad
      idText = idText[:self.max_nr_words]
      
      if use_characters:
        characters = [] 
        characters.extend(line)
        characters = characters[:self.max_nr_chars]
        characters = list(map(lambda x: self.dict_char2id[x], characters))
        characters = characters + [self.dict_char2id['\0']] * (self.max_nr_chars - len(characters))
        idText = idText + characters

      tokens.append(idText)

    return np.array(tokens)



  def input_word_tokens_to_text(self, list_tokens):
    list_tokens = np.array(list_tokens)
    text = []
    rows, cols = list_tokens.shape

    for r in range(rows):
      row_text = ''
      for c in range(cols):
        if self.dict_id2word[list_tokens[r][c]] != '<PAD>':
          row_text = row_text + ' ' + self.dict_id2word[list_tokens[r][c]]
      text = text + [row_text]
    return " ".join(text)
  