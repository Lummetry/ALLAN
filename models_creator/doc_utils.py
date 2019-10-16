import os
import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import json
import random

class DocUtils():
  def __init__(self, logger, gensim_i2v, max_nr_words=None, max_nr_chars=None,
               dict_user_id2label={}, dict_bot_id2label={}):
    self.logger = logger
    self.gensim_i2v = gensim_i2v
    
    with open(self.gensim_i2v, 'rb') as handle:
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
    
    self.dict_user_id2label = dict_user_id2label
    self.dict_user_label2id = {v:k for k,v in self.dict_user_id2label.items()}
    
    self.dict_bot_id2label = dict_bot_id2label
    self.dict_bot_label2id = {v:k for k,v in self.dict_bot_id2label.items()}

    self.all_labels = {}
    
    self.end_char_id = self.dict_word2id['<END>']
    self.start_char_id = self.dict_word2id['<START>']
    
    self.max_nr_words = max_nr_words
    self.max_nr_chars = max_nr_chars
    
    
    self.starts = [
      "Hei, ma numesc Oana, te pot ajuta cu ceva?",
      "Bine ai venit! Numele meu este Oana si m-as bucura sa te pot ajuta cu ceva.",
      "Buna! Ma numesc Oana. Cu ce te pot ajuta?",
      "Buna! Numele meu este Oana. In ce fel as putea sa te ajut?",
      "Buna! Eu sunt Oana si sunt aici sa iti vin in ajutor."
    ]
    
    self.validate = False
  
    return
  
  def _log(self, str_msg, results = False, show_time = False, noprefix=False):
    self.logger.VerboseLog(str_msg, results=results, show_time=show_time,
                           noprefix=noprefix)
    return

  def prepare_for_tokenization(self, string):
    return re.sub(r'([ \w]*)([!?„”"–,\'\.\(\)\[\]\{\}\:\;\/\\])([ \w]*)', r'\1 \2 \3', string)


  def strip_html_tags(self, string):
    return re.sub(r'<.*?>', '', string)


  def _create_labels_vocab(self, fn):
    with open(fn, 'rt') as handle:
      labels = handle.read().splitlines()

    dict_label2id = {labels[i]: i for i in range(len(labels))}
    dict_id2label = {v:k for k,v in dict_label2id.items()}
    
    return dict_label2id, dict_id2label

  def CreateUserLabelsVocab(self, fn):
    self.dict_user_label2id, self.dict_user_id2label = self._create_labels_vocab(fn)

  def CreateBotLabelsVocab(self, fn):
    self.dict_bot_label2id, self.dict_bot_id2label = self._create_labels_vocab(fn)

  
  def GenerateLabels(self, path):
    assert self.dict_bot_label2id != {}
    assert self.dict_user_label2id != {}
    self.all_labels = {}
    for file in os.listdir(path):
      full_path = os.path.join(path, file)
      new_labels = []
      with open(full_path, 'rt') as handle:
        labels = handle.read().splitlines()
        try:
          for i,x in enumerate(labels):
            if i % 2 == 0:
              new_labels.append(self.dict_bot_label2id[x])
            else:
              new_labels.append(self.dict_user_label2id[x])
        except Exception as e:
          print(e)
          print(full_path)
          raise Exception

      self.all_labels[file] = new_labels
    
    self._log("All labels from [{}] were processed.".format(path[-50:]))
    return
  
  def GenerateBatches(self, path, use_characters=True, use_labels=True, eps_words=10, eps_characters=30):
    str_log = "Generating TRAINING batches based on conversations extracted from [..{}] "
    str_log += "(use_characters={}  use_labels={}  eps_words={}  eps_characters={}) ..."
    self._log(str_log.format(path[-50:], use_characters, use_labels, eps_words, eps_characters))
    
    conversations_w, conversations_c = self.tokenize_conversations(path=path,
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
          batches.append((new_corpus, target, current_labels[:(num+1)])) # current_labels contain also the label for the bot
    
    return batches
  
  
  def GenerateValidationBatches(self, path, use_labels=True):
    str_log = "Generating VAL batches based on conversations extracted from [..{}] "
    str_log += "(use_labels={}) ..."
    self._log(str_log.format(path[-50:], use_labels))

    conversations_lines, conversations_labels, conversations_possibilities = self.tokenize_validation_conversations(path)
    
    keys = conversations_lines.keys()
    batches = {}
  
    for k1 in keys:
      for k2 in conversations_lines[k1].keys():
        current_lines = conversations_lines[k1][k2]
        current_possibilities = conversations_possibilities[k1][k2]
        
        if use_labels:
          current_labels = conversations_labels[k1][k2]
        
        assert len(current_lines) == len(current_labels)

        final_key = (k2 + 1) * 2
        if final_key not in batches: batches[final_key] = []
        
        if not use_labels:
          batches[final_key].append((current_lines, current_possibilities))
        else:
          batches[final_key].append((current_lines, current_possibilities, current_labels))
      #endfor
    #endfor

    return batches
    
        
  def tokenize_validation_conversations(self, path):
    conversations_lines = {}
    conversations_possibilities = {}
    conversations_labels = {}
    for file in os.listdir(path):
      if not file.endswith('.json'): continue
      
      conversations_lines[file] = {}
      conversations_possibilities[file] = {}
      conversations_labels[file] = {}
    
      with open(os.path.join(path, file), 'r') as f:
        crt_json = json.load(f)

      crt_conversation_lines = []
      crt_conversation_labels = []
      crt_conversation_possibilities = []
      turns = crt_json['TURNS']
      for i in range(len(turns)):
        crt_turns = []
        crt_labels = []
        crt_possibilities = []

        t = turns[:i+1]

        crt_turns.append(random.choice(self.starts))
        crt_labels.append(self.dict_bot_label2id["salut"])
        
        for j,x in enumerate(t):
          crt_turns.append(x['STATEMENT'])
          crt_labels.append(self.dict_user_label2id[x['LABEL']])
          
          if j != len(t) - 1:
            possibility = random.choice(x['POSSIBILITIES'])
            crt_turns.append(possibility['STATEMENT'])
            crt_labels.append(self.dict_bot_label2id[possibility['LABEL']])
          #endif
        #endfor
        
        crt_possibilities = {
            'STATEMENTS': [self.prepare_for_tokenization(p['STATEMENT']).split() for p in x['POSSIBILITIES']],
            'LABEL' :  self.dict_bot_label2id[x['POSSIBILITIES'][0]['LABEL']] # labels are the same for a group of possibilities
        }

        crt_conversation_lines.append(crt_turns)
        crt_conversation_labels.append(crt_labels)
        crt_conversation_possibilities.append(crt_possibilities)
      #endfor
      
      assert len(crt_conversation_lines) == len(crt_conversation_labels) == len(crt_conversation_possibilities)
      
      for i in range(len(crt_conversation_lines)):
        conversations_lines[file][i] = crt_conversation_lines[i]
        conversations_possibilities[file][i] = crt_conversation_possibilities[i]
        conversations_labels[file][i] = crt_conversation_labels[i]
    #endfor

    return conversations_lines, conversations_labels, conversations_possibilities
  
  
  def tokenize_single_conversation(self, lines, append_to_distributions=False):
    current_conversation_w = []
    current_conversation_c = []
   
    for line in lines:
      if line == '\n': continue
      if line[-1] == '\n': line = line[:-1]
      characters = []
      characters.extend(line)
      if append_to_distributions: self.num_chars_distribution.append(len(characters))

      line_mod = self.prepare_for_tokenization(line)
      line_mod = line_mod.split()
      if append_to_distributions: self.num_words_distribution.append(len(line_mod))
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
    return current_conversation_w, current_conversation_c
      

  def tokenize_conversations(self, path, eps_words=10, eps_characters=30):
    self.num_chars_distribution = []
    self.num_words_distribution = []
    conversations_w = {}
    conversations_c = {}
    self.unknown_words = {}

    for file in os.listdir(path):
      full_path = os.path.join(path, file)
      with open(full_path, 'r') as f:
        lines = f.readlines()
      
      current_conversation_w, current_conversation_c = self.tokenize_single_conversation(lines, append_to_distributions=True)
      
      conversations_c[file] = current_conversation_c
      conversations_w[file] = current_conversation_w
    #endfor

    self.num_chars_distribution = np.array(self.num_chars_distribution).reshape(-1,1)
    self.num_words_distribution = np.array(self.num_words_distribution).reshape(-1,1)

    df_distrib = pd.DataFrame(data=np.concatenate((self.num_words_distribution, self.num_chars_distribution), axis=1),
                              columns=['Words', 'Characters']).describe()

    self._log("Distributions descriptions:\n" + df_distrib.to_string())

    self.max_nr_words = int(df_distrib.loc['max']['Words']) + eps_words
    self.max_nr_chars = int(df_distrib.loc['max']['Characters']) + eps_characters

    self._log("Setting max nr. words to {}".format(self.max_nr_words))
    self._log("Setting max nr. chars to {}".format(self.max_nr_chars))
    
    return conversations_w, conversations_c
    
    
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
    
    text = text[1:]
    return self.organize_text(" ".join(text))

  
  def translate_tokenized_input(self, _input):
    all_sentences = []
    for idx, sentence in enumerate(_input):
      new_sentence = [self.dict_id2word[word] for i,word in enumerate(sentence) if (i < self.max_nr_words) and (self.dict_id2word[word] != '<PAD>')]
      all_sentences.append(new_sentence)
    
    final_str = ""
    for s in all_sentences:
      final_str += " ".join(s)
      final_str += "\n"
    return self.organize_text(final_str)
  
  
  def organize_text(self, text):
    text = text.replace(' ?', '?')
    text = text.replace(' !', '!')
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    return text
  
  def SetPredictionBatches(self, batches_train_to_validate, batches_validation):
    self.validate = True
    self.batches_train_to_validate = batches_train_to_validate
    self.batches_validation = batches_validation
    return
  
  
  def translate_generator_sentences(self, batch, translate_out_batch=True):
    in_batch  = batch[0]
    out_batch = batch[1]
    in_label_batch = None
    if len(batch) == 3:
      in_label_batch = batch[2]
      assert self.dict_bot_id2label is not None and self.dict_user_id2label is not None
    
    # print input
    for idx, sentence in enumerate(in_batch):
        new_sentence = [self.dict_id2word[word] for i,word in enumerate(sentence) if (i < self.max_nr_words) and (self.dict_id2word[word] != '<PAD>')]
        
        if len(batch) == 3:
          if idx % 2 == 0:
            label = self.dict_bot_id2label[in_label_batch[idx]]
          else:
            label = self.dict_user_id2label[in_label_batch[idx]]
          self._log(self.organize_text(" ".join(new_sentence)) + ' -> ' + label, noprefix=True)
        else:
          self._log(self.organize_text(" ".join(new_sentence)), noprefix=True)
    
    # print output
    if translate_out_batch:
      new_sentence = [self.dict_id2word[word] for word in out_batch]
      self._log(self.organize_text(" ".join(new_sentence)) + ' -> ' + self.dict_bot_id2label[in_label_batch[-1]], noprefix=True)
    return