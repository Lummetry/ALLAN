# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:14:55 2019

@author: Andrei
"""

from libraries.logger import Logger
from libraries.model_server.simple_model_server import SimpleFlaskModelServer

from tagger.brain.base_engine import ALLANTaggerEngine

import numpy as np

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
  

  #####      
  # now load FlaskServer tools and .run()
  #####
  def input_callback(data):
    if 'current_query' not in data.keys():
      return None
    s = data['current_query']
    return s
  
  def output_callback(data):
    res1 = data[0]
    inputs = data[1]
    std_input = inputs[0]
    enc_input = inputs[1]
    vals = [x for x in res1.values()]
    keys = [x for x in res1.keys()]    
    top_idxs = np.argsort(vals)[::-1]
    dic_top = {keys[x]:float(round(vals[x],3)) for x in top_idxs[:3]}
    dic_non_top = {keys[x]:float(round(vals[x],3)) for x in top_idxs[3:]}
    id_topic_document = -1 if DEBUG_MODE == 0 else data[2]
    
    dct_info = {
        'general_tags' : dic_non_top, 
        'top_tags': dic_top, 
        'input_document_init' : std_input,
        'input_document_post' : enc_input, 
        'id_topic_document': id_topic_document,
        'comment' : 'id_topic_document == -1 means ALLANTagger is in DEBUG(0) mode => no topics are available. Switch to DEBUG(1) or NODEBUG.'
        }
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
  # http://127.0.0.1:5001/current_query?input_text=ala+bala+portocala
    
