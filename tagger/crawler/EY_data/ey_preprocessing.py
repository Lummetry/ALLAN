import json
import pickle
import random
import pandas as pd

from collections import Counter
from libraries.logger import Logger
from nltk.tokenize import TweetTokenizer

random.seed(1)

def flatten_list(a):
  return [item for sublist in a for item in sublist]

class EY_Data(object):  
  """
  Object for preprocessing raw data of the form: questions, answers, topics (each in different files)
  into training + validation dataset for a document tagger.
  
  This object constructs the dataset by using the entire answer and each line in the questions file as individual observations,
  and each observation has a topic label, which is drawn from the respective topic file.
  For each topic there are a number of additional generated labels from the text of all observations in the topic, 
  and for each observation only the labels which can also be found in the text of the observation is kept.
  
  Generating labels in this manner means dealing with a large volume of tags that add noise to the tagger's training data.
  To remove these undesirable tags that hinder the predictive power of the tagger, a list of exclusions(labels automatically removed
  from the dataset), and a dictonary of reductions(labels that are replaced by other labels) are added through a json.
  
  This object can: generate the dataset described above and write them to disk,
                   generate the topic label mapping and write it to disk,
                   process the EY feedback as returned by the tagger API.
                   
  """
  def __init__(self, logger, folder_name=None, index_of_last_file=None, occurence_threshold=None, validation_set_length=None):
    self.logger = logger
    
    self.folder_name = folder_name
    self.index_of_last_file = index_of_last_file
    self.occurence_threshold = occurence_threshold
    self.validation_set_length = validation_set_length
    
    # _data folder in the app folder of the Lummetry project
    self._data_app_folder = self.logger.GetDropboxDrive() + '/' + self.logger.config_data['APP_FOLDER'] + '/_data/'
    
    self._parse_config_data()
    
    self._parse_json_data()
    
    self.tokenizer = TweetTokenizer()
    
    #folder containing data for preprocessing
    self.data_dir = self._data_app_folder + self.folder_name

    self.read_files()

    self.generate_raw_labels()
    
    self.process_labels()
    
    self.build_topic_label_map()
    self.build_tag2idx_map()
    
  def _parse_config_data(self):
    """
    Method for parsing the parameters passed through the config file.
    
    The parameters are:
      
      folder_name: the folder name where the raw data from EY is located(format:topics, questions, answers)
      
      index_of_last_file: the index of the last file in the raw data, so the read knows when to stop
      
      occurence_threshold: float between 0-1,
                           each label occurs in a percentage of the total final observations, if the percentage of a label
                           is above this threshold, the label is removed from the dataset.
                           
      validation_set_length: the number of observations reserved for the validation set
    """
    if self.folder_name is None:
      self.folder_name = self.logger.config_data['DATA_FOLDER_NAME']
      
    if self.index_of_last_file is None:
      self.index_of_last_file = self.logger.config_data['INDEX_OF_LAST_FILE']
      
    if self.occurence_threshold is None:
      self.occurence_threshold = self.logger.config_data['OCCURENCE_THRESHOLD']
      
    if self.validation_set_length is None:
      self.validation_set_length = self.logger.config_data['VALIDATION_SET_LENGTH']
    
    return
  
  def _parse_json_data(self):
    """
    Method for parsing the parameters of the object passed through JSON.
    
    The parameters are: 
      self.undesirable_list: a list of labels to be automatically excluded from the dataset
      
      self.dict_lbl_simpler: a dictionary where keys are labels and values are lists of labels
                             if a label is found in the values of this list, it is replaced with its key
    """
    with open(self._data_app_folder + 'exclusions.json', 'r', encoding='utf-8') as f:
      self.undesirable_list = json.load(f)
      
    with open(self._data_app_folder + 'reductions.json', 'r', encoding='utf-8') as f:
      self.dict_lbl_simpler = json.load(f)  
    
    return
    
  def read_files(self):
    """
    Reads answers, questions and topics files, from the data directory.
    
    Arguments: number of answer, question, topic files
    
    Populates the answers, questions and topic lists of the EY_Data object
    with the content of the files.
    """
    self.answers = []
    self.questions = []
    self.topics = []
    
    for i in range(self.index_of_last_file):
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
    
    return
  
  def label_information(self, labels):
    """
    Prints important metrics concerning the labels list it is fed.
    
    Arguments: a list of lists of labels. each observation in the dataset
               contains a list of labels. The function needs the list of lists of labels
               
    Prints: label frequency,
            number of topic_labels,
            distribution of lengths of label list in dataset,
            labels as grouped by their frequency,
            pandas describe distribution of labels.
    
    """
    #update flattened labels
    flattened_labels = flatten_list(labels)
    #update label occurence counter
    dict_label_occurence = Counter(flattened_labels)
    self.logger.P('Word frequency in labels:\n {}'.format(dict_label_occurence))
    self.logger.P('Number of topic labels:')
    for k,v in dict_label_occurence.items():
      if len(k) > 5:
        if k[:5] == 'topic':
          self.logger.P('{}:{}'.format(k,v))

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
    """
    Constructs a list of tags based on a line(sentence).
    uses NLTK tokenizer to generate tokens, 
    each token is added to the list of tags 
    IF the token is not in self.undesirable_tags
       and the token is of length larger than 2
    
    
    Arguments: line - a sentence(could have a single word)
    """
    tags = []
    tokenized_line = self.tokenizer.tokenize(line)
    while '' in tokenized_line:
      tokenized_line.remove('')
    for i in tokenized_line:
      if len(i) > 2 and i not in self.undesirable_list:
        tags.append(i)
    
    return tags
      
  def generate_raw_labels(self):
    """
    Generates a list of lists of raw labels.
    Each list of labels is associated to an answer, topic, question pair.
    The raw labels are generated using the line_to_tags function.
    
    Another list of labels is generated soley from the topics.
    These are unique to each label and are denoted by topic_"topic name"
    
    """
    self.labels = []
    self.topic_labels = []
    for i in range(len(self.questions)):
      lbl = []
      qs = self.questions[i].splitlines()
      for j in qs:
        lbl.append(self.line_to_tags(j))
            
      unique_topic_label = "topic_" + "_".join([x[:8].replace('-','') for x in self.topics[i].split(' ') if len(x)>1])
      self.topic_labels.append(unique_topic_label)
       
      lbl = flatten_list(lbl)
      lbl = list(set(lbl))
      self.labels.append(lbl)
      
    return
  
  def generate_dict_label_occ_in_docs(self, labels):
    """
    Generates a dictionary containing the occurence of each label in
    the final dataset.
    
    Arguments: a list of lists of labels
    
    Returns: a dictionary where keys are tags, and values are their occurence values
    
    """
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
    """
    A function meant to remove a single tag from all sublists of
    the self.labels list of lists.
    
    Arguments: an item that can be found in self.labels
    
    """
    self.logger.P('Removing tag {} from all labels...'.format(tag))
    new_labels = []
    for i in self.labels:
      new_labels.append(list(filter(lambda a: a != tag, i)))
      
    self.labels = new_labels
    
    return

  def replace_tags(self, replacement_tag, tag_to_replace):
    """
    Replaces a single tag with a different tag in self.labels
    
    Arguments: tag_to_replace: the label which is going to be replaced
               replacement_tag: the label which is going to replace
    """
    new_labels = []
    for row in self.labels:
      new_row = []
      if tag_to_replace in row:
        for item in row:
          if item == tag_to_replace:
            new_row.append(replacement_tag)
          else:
            new_row.append(item)
      else:
        new_row = row
        
      new_row = list(set(new_row))
      new_labels.append(new_row)
      
    self.labels = new_labels
    
  def replace_tags_in_row(self, row, replacement, tag_to_replace):
    """
    Replaces a single tag in a single list of labels
    
    Arguments: row: list of labels
               replacement: label which will replace 
               tag_to_replace: label which will be replaced
               
    Returns a list of labels where the replacement is applied
    """
    new_row = []
    for item in row:
      if item == tag_to_replace:
        new_row.append(replacement)
      else:
        new_row.append(item)
    
    return new_row
    
  def process_labels(self):
    """
    Function that removes labels that occur in a large enough percentage of the observations
    
    Meant to remove noise from the distribution of labels, with the assumption that
    the most meaningful tags are the ones that occur less frequently.
    
    The percentage of occurence of each tag is computed and if it is above the
    occurence threshold as initialized, it will be removed from all lists of labels.
    """
    self.flattened_labels = flatten_list(self.labels)

    self.dict_label_frequency = Counter(self.flattened_labels)
    self.common_labels = []
    
    self.dict_label_occurence = self.generate_dict_label_occ_in_docs(self.labels)
    self.logger.P('dict label occurence: \n {}'.format(self.dict_label_occurence))
    
    self.label_information(self.labels)

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
    
    self.logger.P('Removing tags from undesirable list...')
    for i in self.undesirable_list:
      self.remove_tag_from_labels(i)
    
    self.dict_label_occurence = self.generate_dict_label_occ_in_docs(self.labels)
    self.logger.P('dict label occurence: \n {}'.format(self.dict_label_occurence))
    
    self.flattened_labels = flatten_list(self.labels)
      
    return
  
  def build_topic_label_map(self, write_to_disk=False):
    """
    Builds a mapping(dictionary) between topic_labels, and all other labels.
    
    The topic_labels are unique to each topic/question/answer pairs, but the other
    labels are not always unique to a single topic. The function builds a 
    dictionary that visualizes that mapping.
    
    The dictionary is written do disk as it is needed by the document tagger 
    for training.
    
    Arguments: write_to_disk: boolean indicating whether the mapping will be 
                              written to a pickle file.
    """
    self.topic_label_map = dict.fromkeys(self.topic_labels, [])
    labels_copy = self.labels
    topic_label_map_values = []
    for label_row in labels_copy:
      for k,v in self.dict_lbl_simpler.items():
        for tag in label_row:
          if tag in v:
            label_row = self.replace_tags_in_row(label_row, k, tag)
      topic_label_map_values.append(list(set(label_row)))
      
    self.logger.P('Mapping of tags to topics:')
    for index in range(len(self.topic_labels)):
      self.topic_label_map[index] = [self.topic_labels[index]] + topic_label_map_values[index]
    
    if write_to_disk:
      with open(self._data_app_folder + 'topic_tag_map_v3.pkl','wb') as file:
        pickle.dump(self.topic_label_map, file)
      
      self.build_tag2idx_map(write_to_disk=True)
        
  def build_tag2idx_map(self, write_to_disk=False):
    """
    Creates a dictionary indexing the labels.
    
    Each label is attributed a number, number which is incremented atomically.
    
    It relies on the self.topic_label_map object which is intialized once the 
    self.build_topic_label_map is run. So make sure that function is called before
    calling this one!
    
    The mapping is printed on the screen.
    """
    dic_lbl2idx = {}
    index = 0 
    try:
      for tags in self.topic_label_map.values():
        for tag in tags:
          if tag not in list(dic_lbl2idx.keys()):
            dic_lbl2idx[tag] = index
            index += 1
        
      for k,v in dic_lbl2idx.items():
        print('{} : "{}"'.format(k,v))
        
      if write_to_disk:  
        with open(self._data_app_folder + 'topic_tag2idx_v3.txt',"w") as f:
          f.write("{\n")
          for k,v in dic_lbl2idx.items():
            f.write('{} : "{}" \n'.format(k,v))
          f.write("}")
        
    except AttributeError as e:
      self.logger.P('[ERROR] self.topic_label_map is not initialized. Please call the build_topic_label_map function before calling this function!')
      raise type(e)('[ERROR] self.topic_label_map is not initialized. Please call the build_topic_label_map function before calling this function!')
      
  def intersect_text_and_labels(self, text, labels):
    """
    Function that intersects the text of an observation with the full list of 
    tags for the topic/question/answer pair that the observation is a part of.
    
    Additionally, labels are simplified by using the self.dict_lbl_simpler mapping
    
    Arguments: text: a string(sentence) that is a part of the dataset
               labels: the whole list of tags of the topic 
    
    Returns: a smaller list of labels populated by the:
             topic_label of the observation, and any additional tags that from 
             the whole list of labels that also appears in the text
    """
    found_labels = []
    found = 0
    flattened_values = flatten_list(list(self.dict_lbl_simpler.values()))
    tokenized_text = self.tokenizer.tokenize(text)
    while '' in tokenized_text:
      tokenized_text.remove('')
    for i in labels:
      if i in tokenized_text:
        if i in flattened_values:
          found = 0
          while not found:
            for k,v in self.dict_lbl_simpler.items():
              if i in v:
                found_labels.append(k)
                found = 1
        else:
          found_labels.append(i)

    found_labels = list(set(found_labels))
    return found_labels
  
  def build_outputs(self):
    """
    Function that builds the dataset.
    Iterates through topic/questions/answers set, splits questions into individual 
    observations.
    
    Builds a training set and a validation set of size: self.validation_set_length
    Each observation initially contains the full labels for the topic, but its 
    label list gets reduced with intersect_text_and_labels.
    
    Returns: output_texts: list of text observations
             output_labels: list of labels for each text
             validation_texts: list of validation texts
             validation_labels: list of  labels for each validation text
    
    """
    validation_index = 0
    output_texts = []
    output_labels = []
    validation_texts = []
    validation_labels = []
    for i in range(len(self.answers)):
      qs = self.questions[i].splitlines()
      self.logger.P('======= INDEX {} ======'.format(i))
      #populate validation set with random questions from each topic
      if validation_index < self.validation_set_length:
        random_q_idx = random.randint(0,len(qs) - 1)
        validation_texts.append(qs[random_q_idx])
        #construct list of labels
        label_list = self.intersect_text_and_labels(qs[random_q_idx], self.labels[i])
        label_list.append(self.topic_labels[i])
        label_list = list(set(label_list))
        
        validation_labels.append(label_list)
        #delete question used for validation set from training set
        del qs[random_q_idx]
        validation_index += 1
      #remove empty questions
      while '' in qs:
        qs.remove('')
      #build each question with its labels
      for j in qs:
        self.logger.P(j)
        output_texts.append(j)
        #only choose relevant labels
        lbl = self.intersect_text_and_labels(j, self.labels[i])
        lbl.append(self.topic_labels[i])
        lbl = list(set(lbl))
        self.logger.P(str(lbl))
        output_labels.append(lbl)
      
      #build answers with all labels
      output_texts.append(self.answers[i])
      self.logger.P(self.answers[i])
      #only choose relevant labels
      lbl = self.labels[i]
      for k,v in self.dict_lbl_simpler.items():
        for tag in lbl:
          if tag in v:
            lbl = self.replace_tags_in_row(lbl, k, tag)
            
      lbl.append(self.topic_labels[i])
      lbl = list(set(lbl))
      self.logger.P(str(lbl))
      output_labels.append(lbl)
      
    self.label_information(output_labels)  
    return output_texts, output_labels, validation_texts, validation_labels
  
  def write_to_file(self):
    """
    Writes to file the training and validation dataset.
    
    """
    dir_location = self.logger.GetDropboxDrive()  + '/' + self.logger.config_data['APP_FOLDER']
    texts, labels, valid_texts, valid_labels = self.build_outputs()
    self.logger.P('================ information on output labels ================')
    self.label_information(labels)
    self.logger.P('================ end information on output labels ================')
    #write training_data
    for i in range(len(texts)):
      fname = 10000 + i
      fname = str(fname)[1:]
      with open(dir_location + '/_data/EY_FAQ/v2/Texts/Text_%s.txt' % fname, 'w', encoding='utf-8') as f_doc:
        f_doc.write(texts[i])
      f_doc.close()

      with open(dir_location + '/_data/EY_FAQ/v2/Labels/Text_%s.txt' % fname, 'w', encoding='utf-8') as f_label:
        for j in labels[i]:
          f_label.write(j + '\n')
      f_label.close()
      
      #write validation_data
      if i < len(valid_texts):
        with open(dir_location + '/_data/EY_FAQ/v2/Validation_Texts/Text_%s.txt' % fname, 'w', encoding='utf-8') as f_doc:
          f_doc.write(valid_texts[i])
        f_doc.close()
  
        with open(dir_location + '/_data/EY_FAQ/v2/Validation_Labels/Text_%s.txt' % fname, 'w', encoding='utf-8') as f_label:
          for k in valid_labels[i]:
            f_label.write(k + '\n')
        f_label.close()

  def get_entries(self, fn_questions):
    """
    Parses the txt files with tests returned by EY
    
    Arguments: fn_questions: filename of the document containing the texts
    
    Returns: a list of entries where an entry is the string 
             between curly brackets containing the information of the tagger's 
             return
    """
    folder = self.data_dir[:-1].rsplit('/',1)[0]
    list_of_entries = []
    entry = []
    with open(folder + fn_questions) as file:
      content = file.readlines()
      for line in content:
        line = line.strip()
        if len(line) >0 and line[0] == '"':
          entry = []
          entry.append(line[1:])
        elif len(line) > 0 and line[-3] == '}':
          entry.append(line[:-2])
          list_of_entries.append(entry)
        else:
          entry.append(line)
    return list_of_entries
  
  def parse_score(self, score):
    """
    A function that cleans and turns the scores for a label into float
    
    Since the EY testing team marked "XX" wherever the tagger was wrong,
    to get the actual values returned by the tagger it is necessary to remove
    any additional string.
    
    Returns a float score of the label
    """
    if 'X' in score:
        score = score.strip('X')
    if '}' in score:
      score = score.strip('}')
    
    score = float(score)
    return score
    
  
  def get_tags(self, entry):
    """
    Processes an entry and returns its information.
    
    Arguments: a single entry from the list of entries as returned by get_entries
    
    Returns: proc_query: the processed query of the entry(what allan sees)
             query: the unprocessed query of the entry(raw input)
             tags: a list of tuples containing label name and float score
    """
    self.flattened_labels = flatten_list(self.labels)
    tags = []
    for line in entry:
      if line[:4] == 'resu':
        result_line_split =  line.split('{')[-1].split('"')
        tag = result_line_split[-3]
        conf = result_line_split[-1]
        conf = conf[1:6]
        conf = self.parse_score(conf)
        tag_score = (tag,conf)
        tags.append(tag_score)
      elif line[:16] == 'input_document_i':
        query = line[18:].split('"')[2]
      elif line[:16] == 'input_document_p':
        proc_query = line[18:].split('"')[2]
      else:
        first_word = line.split(':')
        if first_word[0] in self.flattened_labels or first_word[0] in self.topic_labels or first_word[0] in list(self.dict_lbl_simpler.keys()):
          conf = first_word[1][:5]
          conf = self.parse_score(conf)
          tag_score = (first_word[0], conf)
          tags.append(tag_score)
        if first_word[0][:3] == 'run':
          conf = first_word[2][:5]
          conf = self.parse_score(conf)
          tag_score = (first_word[1].split('"')[2], conf)
          tags.append(tag_score)
    return proc_query, query, tags
  
  def find_topic_in_map(self, topic_map):
    """
    Finds the topic of a topic_map
    
    Arugments: topic_map: a dictionary where keys: topic_labels and values are 
                      the tags that appear in the topic(as in topic_label_map)
    
    Returns: max_len_key: the topic where the list of tags contained is the longest
             max_sum_key: the topic where the sum of list of tags is the largest
             
    """
    max_len_key = ''
    max_sum_key = ''
    topic_identification_len_map = {k: len(v) for k,v in topic_map.items()}
    topic_identification_sum_map = {k: 0 for k in topic_map.keys()}
    
    for key, values in topic_map.items():
      topic_identification_sum_map[key] += sum([pair[1] for pair in values])
    
    max_len_key = max(topic_identification_len_map, key=lambda k: topic_identification_len_map[k])
    max_sum_key = max(topic_identification_sum_map, key=topic_identification_sum_map.get)

    self.logger.P('The identifcation maps:')
    for k,v in topic_map.items():
      if len(v) > 0:
        self.logger.P('{}: length = {} sum = {} -- {}'.format(k, topic_identification_len_map[k], topic_identification_sum_map[k], v))
    
    return max_len_key, max_sum_key


  def best_topic(self, entry):
    """
    Function that generates topic_map of an entry and returns final topic
    
    Arguments: a single entry from the list of entries as returned by get_entries
    
    Generates a topic_map: a dictionary where keys: topic_labels and values are 
    the tags that appear in the topic(as in topic_label_map), and applies 
    find_topic_in_map on the generated topic_map
    
    Prints valuable information for the entry: query, proc_query(what allan sees),
    max len topic, max sum topic.
    """
    proc_query, query, tags = self.get_tags(entry)
    print()
    topic_identification_map = {k:list() for k in self.topic_labels}
    #topic identification on best tags:
    for best_tag in tags:
      if best_tag[0] in self.topic_labels:
        for topics in self.topic_labels:
          if best_tag[0] == topics:
            topic_identification_map[topics].append(best_tag)
      found_in_topics = []
      for keys, tag_lists in self.topic_label_map.items():
        for tag in tag_lists:
          if tag == best_tag[0]:
            found_in_topics.append(keys)

      for topic in found_in_topics:
        topic_identification_map[topic].append(best_tag)

    self.logger.P('Query: {}'.format(query), noprefix=True)
    self.logger.P('PostQuery: {}'.format(proc_query), noprefix=True)
    max_len_topic, max_sum_topic = self.find_topic_in_map(topic_identification_map)
    self.logger.P('Max length topic : {}'.format(max_len_topic), noprefix=True)
    self.logger.P('Max sum topic : {}'.format(max_sum_topic), noprefix=True)
    
    return 

if __name__ == '__main__':
  
  logger = Logger(lib_name='EY_DATA',
                  config_file='./tagger/crawler/EY_data/config_eydata.txt',
                  TF_KERAS=False)
  
  data = EY_Data(logger)
  data.build_outputs()
  #uncomment next line to write built dataset to disk
#  data.write_to_file()
#  data.build_topic_label_map(write_to_disk=True)

  #UNCOMMENT THIS BIT TO GET RESUTLS OF EY tests, change file argument for new tests
#  list_of_entries = data.get_entries('/ALLEN_Wrong_Questions_2.txt')
#
#  for entry in list_of_entries:
#    data.best_topic(entry)
