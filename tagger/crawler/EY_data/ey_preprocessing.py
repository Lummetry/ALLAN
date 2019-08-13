import random
import pandas as pd

from collections import Counter
from libraries.logger import Logger
from nltk.tokenize import TweetTokenizer

random.seed(1)

def flatten_list(a):
  return [item for sublist in a for item in sublist]

class EY_Data(object):  
  def __init__(self, logger, folder_name, occurence_threshold=0.25, validation_set_length=10):
    self.logger = logger
    self.folder_name = folder_name
    self.occurence_threshold = occurence_threshold
    self.validation_set_length = validation_set_length
    
    self.tokenizer = TweetTokenizer()

    self.data_dir = logger.GetDropboxDrive() + '/' + logger.config_data['APP_FOLDER'] + self.folder_name
    
    self.read_files()
    
    self.undesirable_list = ['']
    
    self.generate_raw_labels()
    
    self.process_labels()
 
#    self.build_outputs()
#    self.write_to_file()
    
  def read_files(self):
    self.answers = []
    self.questions = []
    self.topics = []
    
    for i in range(25):
      logger.P('Processing file {}...'.format(i))
      
      a_filename='a' + str(i + 1) + '.txt'
      q_filename='q' + str(i + 1) + '.txt'
      t_filename='t' + str(i + 1) + '.txt'
      
      a_file = open(self.data_dir + a_filename, 'r', encoding='utf-8', errors='ignore')
      q_file = open(self.data_dir + q_filename, 'r', encoding='utf-8', errors='ignore')
      t_file = open(self.data_dir + t_filename, 'r', encoding='utf-8', errors='ignore')
      
      answer = a_file.read()
      question = q_file.read()
      topic = t_file.read()
      
      self.answers.append(answer.lower())
      self.questions.append(question.lower())
      self.topics.append(topic.lower())
  
  def label_information(self, labels):
    #update flattened labels
    flattened_labels = flatten_list(labels)
    #update label occurence counter
    dict_label_occurence = Counter(flattened_labels)
    
    self.logger.P('Word frequency in labels:\n {}'.format(dict_label_occurence))

    inv_dict_label_occurence = {}
    for k, count in dict_label_occurence.items():
      try:
        inv_dict_label_occurence[count].append(k)
      except KeyError:
        inv_dict_label_occurence[count] = [k]

    dict_label_count = {}
    total_count = 0
    for k, v in inv_dict_label_occurence.items():
      length = len(v)
      dict_label_count[k] = [length]
      total_count += k * length

    lengths_of_labels = []
    for i in range(len(labels)):
      lengths_of_labels.append(len(labels[i]))

    df_lbl_len = pd.DataFrame(columns=['len'])
    df_lbl_len.len = lengths_of_labels
    df_lbl_len_distrib = df_lbl_len.describe()

    self.logger.P('The distribution of lengths of labels for each document: \n {}'.format(df_lbl_len_distrib.to_string()))
    self.logger.P('Length of flattened labels array {} must be equal to added values in dict of word lengths {}'.format(len(flattened_labels), total_count))
    self.logger.P('Labels grouped by frequency \n {}'.format(dict_label_count))

    df = pd.DataFrame(columns=['labels'])
    df.labels = flattened_labels
    df_distrib = df.describe(include='all')

    self.logger.P("Distribution of labels: \n {}".format(df_distrib.to_string()))

    return

  def line_to_tags(self, line):
    tags = []
    tokenized_line = self.tokenizer.tokenize(line)
    while '' in tokenized_line:
      tokenized_line.remove('')
    for i in tokenized_line:
      if len(i) > 2:
        tags.append(i)
    
    return tags
      
  def generate_raw_labels(self):
    self.labels = []
    for i in range(len(self.questions)):
      lbl = []
      qs = self.questions[i].splitlines()
      for j in qs:
        lbl.append(self.line_to_tags(j))
        
      lbl.append(self.tokenizer.tokenize(self.topics[i]))

      lbl = flatten_list(lbl)
      lbl = list(set(lbl))
      self.labels.append(lbl)
    
  def generate_dict_label_occ_in_docs(self, labels):
    unqiue_labels  = list(set(flatten_list(labels)))
    dictonary = {}
    for i in labels:
      for k in unqiue_labels:
        if k in i:
          try:
            dictonary[k] += 1
          except:
            dictonary[k] = 1
    
    return dictonary
  
      
  def remove_tag_from_labels(self, tag):
    self.logger.P('Removing tag {} from all labels...'.format(tag))
    new_labels = []
    for i in self.labels:
      new_labels.append(list(filter(lambda a: a != tag, i)))
      
    self.labels = new_labels
      
  def process_labels(self):
    self.flattened_labels = flatten_list(self.labels)
    self.dict_label_frequency = Counter(self.flattened_labels)
    self.common_labels = []
    
    self.dict_label_occurence = self.generate_dict_label_occ_in_docs(self.labels)
    self.logger.P('dict label occurence: \n {}'.format(self.dict_label_occurence))
    
    #REMOVE COMMON WORDS
    for i in self.dict_label_occurence.keys():
      if self.occurence_threshold < 1:
        percentage = self.dict_label_occurence.get(i)/len(self.answers)
        if percentage > self.occurence_threshold:
          self.logger.P('{} word appears in {}% of documents, will be removed from dataset'.format(i, str(percentage * 100)))
          self.common_labels.append(i)
      else:
        if self.dict_label_occurence.get(i) >= self.occurence_threshold:
          self.logger.P('{} word appears in {} of documents, will be removed from dataset'.format(i,self.dict_label_occurence.get(i)))
          self.common_labels.append(i)
  
    self.logger.P('Most common tags that will be removed from list of labels:')
    for i in self.common_labels:
      self.remove_tag_from_labels(i)
    self.logger.P('Total number of removed labels {}'.format(len(self.common_labels)))
    
    self.dict_label_occurence = self.generate_dict_label_occ_in_docs(self.labels)
    self.logger.P('dict label occurence: \n {}'.format(self.dict_label_occurence))
    
    return
  
  def intersect_text_and_labels(self, text, labels):
    found_labels = []
    tokenized_text = self.tokenizer.tokenize(text)
    while '' in tokenized_text:
      tokenized_text.remove('')
    for i in labels:
      if i in tokenized_text:
        found_labels.append(i)
    
    return found_labels
  
  def build_outputs(self):
    validation_index = 0
    output_texts = []
    output_labels = []
    validation_texts = []
    validation_labels = []
    for i in range(len(self.answers)):
      qs = self.questions[i].splitlines()
      #populate validation set with random questions from each topic
      if validation_index < self.validation_set_length:
        random_q_idx = random.randint(0,len(qs) - 1)
        validation_texts.append(qs[random_q_idx])
        validation_labels.append(self.intersect_text_and_labels(qs[random_q_idx], self.labels[i]))
        del qs[random_q_idx]
        validation_index += 1
      #remove empty questions
      while '' in qs:
        qs.remove('')
      for j in qs:
        output_texts.append(j)
        #only choose relevant labels
        lbl = self.intersect_text_and_labels(j, self.labels[i])
        output_labels.append(lbl)
      output_texts.append(self.answers[i])
      #only choose relevant labels
      lbl = self.intersect_text_and_labels(self.answers[i], self.labels[i])
      output_labels.append(lbl)
    
    return output_texts, output_labels, validation_texts, validation_labels
  
  def write_to_file(self):
    dir_location = self.logger.GetDropboxDrive()  + '/' + self.logger.config_data['APP_FOLDER']
    texts, labels, valid_texts, valid_labels = self.build_outputs()
    self.logger.P('================ information on output labels ================')
    self.label_information(labels)
    self.logger.P('================ end information on output labels ================')
    #write training_data
    for i in range(len(texts)):
      with open(dir_location + '/_data/EY_FAQ/Texts/Text_%s.txt' % i, 'w', encoding='utf-8') as f_doc:
        f_doc.write(texts[i])
      f_doc.close()

      with open(dir_location + '/_data/EY_FAQ/Labels/Labels_%s.txt' % i, 'w', encoding='utf-8') as f_label:
        for j in labels[i]:
          f_label.write(j + '\n')
      f_label.close()
      
      #write validation_data
      if i < len(valid_texts):
        with open(dir_location + '/_data/EY_FAQ/Validation_Texts/Text_%s.txt' % i, 'w', encoding='utf-8') as f_doc:
          f_doc.write(valid_texts[i])
        f_doc.close()
  
        with open(dir_location + '/_data/EY_FAQ/Validation_Labels/Labels_%s.txt' % i, 'w', encoding='utf-8') as f_label:
          for k in valid_labels[i]:
            f_label.write(k + '\n')
        f_label.close()
        
if __name__ == '__main__':
  
  logger = Logger(lib_name='EY_DATA',
                  config_file='./tagger/crawler/EY_data/config_eydata.txt',
                  TF_KERAS=False)
  
  data = EY_Data(logger, '/_data/EY_FAQ_RAW/', occurence_threshold=3)