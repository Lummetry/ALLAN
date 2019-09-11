# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:14:55 2019

@author: Andrei
"""

from libraries.logger import Logger
from libraries.model_server.simple_model_server import SimpleFlaskModelServer

from tagger.brain.base_engine import ALLANTaggerEngine

import numpy as np
import pickle as pkl

from datetime import datetime



def find_topic(logger, topic_tag_map, dict_tags, choose_by_length=False):
  """
  Returns the topic id based on the tags found by the document tagger.
  
  First constructs a topic_identification map: a dictionary where all the keys
  are the topics, and the values are lists of the tags which appear in 
  the documents where the topic appears.
  
  Following that, another dictionary is constructed where the keys are the topics 
  but the values are length of the list in the first dictionary/ sum of the probabilities.
  
  Returns the max value of the dictionary as the found topic
  """
  
  topic_labels = list(topic_tag_map.keys())
  topic_identification_map = {k:list() for k in topic_labels}
  #topic identification on best tags:
  for tag, conf in dict_tags.items():
    if tag in topic_labels:
      for topic in topic_labels:
        if tag[0] == topic:
          topic_identification_map[topic].append((tag,conf))
    found_in_topics = []
    for keys, tag_lists in topic_tag_map.items():
      for label in tag_lists:
        if label == tag:
          found_in_topics.append(keys)

    for topic in found_in_topics:
      topic_identification_map[topic].append((tag,conf))

  if choose_by_length:
    topic_identification_len_map = {k: len(v) for k,v in topic_identification_map.items()}
    max_len_key = max(topic_identification_len_map, key=lambda k: topic_identification_len_map[k])
    return max_len_key
  
  else:
    topic_identification_sum_map = {k: 0 for k in topic_identification_map.keys()}
  
    for key, values in topic_identification_map.items():
      topic_identification_sum_map[key] += sum([pair[1] for pair in values])
    
    max_sum_key = max(topic_identification_sum_map, key=topic_identification_sum_map.get)
  
    return max_sum_key

if __name__ == '__main__':
  
  cfg1 = "tagger/brain/configs/config.txt"
  
  use_raw_text = True
  force_batch = True
  use_model_conversion = False
  epochs = 30
  use_loaded = True
  
  l = Logger(lib_name="ALNT",config_file=cfg1, HTML=True)
  l.SupressTFWarn()
  
  eng = ALLANTaggerEngine(log=l,)
  
  eng.initialize()
  
  DEBUG_MODE = 0
  
  TOP = 5
    
  topic_tag_map_path = l.GetDropboxDrive() + '/' + l.config_data['APP_FOLDER'] + '/_data/EY_topic_tag_map.pkl'   
  
  with open(topic_tag_map_path, 'rb') as handle:
    topic_tag_map = pkl.load(handle)

  topic_index_map = {k:0 for k in topic_tag_map.keys()}
  idx = 0
  for k in topic_index_map.keys():
    topic_index_map[k] = idx
    idx += 1
    
  #####      
  # now load FlaskServer tools and .run()
  #####
  def input_callback(data):
    if 'current_query' not in data.keys():
      return None
    s = data['current_query']
    return s
  
  def output_callback(data):
    DEBUG = False
    res1 = data[0]
    inputs = data[1]
    std_input = inputs[0]
    enc_input = inputs[1]
    vals = [x for x in res1.values()]
    keys = [x for x in res1.keys()]    
    top_idxs = np.argsort(vals)[::-1]
    dic_top = {keys[x]:float(round(vals[x],3)) for x in top_idxs}
    dic_top_best ={keys[x]:float(round(vals[x],3)) for x in top_idxs[:TOP]}
    dic_top_runner ={keys[x]:float(round(vals[x],3)) for x in top_idxs[TOP:]}

    topic_document = find_topic(l, topic_tag_map, dic_top, False) #USE True to check by length    
    id_topic_document = topic_index_map[topic_document]
    str_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:22]
    
    
    dct_info = {
        'input_document_init' : std_input,
        'id_topic_document': id_topic_document,
        'topic_document': topic_document,
        'date_time' : str_now,
        }
    
    if DEBUG:
      dct_info['input_document_post'] = enc_input
      dct_info['runner_tags'] = dic_top_runner
      dct_info['best_tags'] = dic_top_best
      dct_info['comment'] = 'id_topic_document == -1 means ALLANTagger is in DEBUG(0) mode => no topics are available. Switch to DEBUG(1) or NODEBUG.'


    dct_res = {'result' : dct_info}
    return dct_res
  
  
  simple_server = SimpleFlaskModelServer(model=eng,
                                         predict_function='predict_text',
                                         fun_input=input_callback,
                                         fun_output=output_callback,
                                         log=l,
                                         host=l.config_data['HOST'],
                                         port=l.config_data['PORT'])
  simple_server.run()
  
  # now we ca run
  # http://127.0.0.1:5001/analyze?current_query=ala+bala+portocala
    
