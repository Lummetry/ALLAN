# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:14:55 2019

@author: Andrei
"""

from libraries.logger import Logger
from libraries.model_server.simple_model_server import SimpleFlaskModelServer

from tagger.brain.base_engine import ALLANTaggerEngine


if __name__ == '__main__':
  
  cfg1 = "tagger/brain/config.txt"
  
  use_raw_text = True
  force_batch = True
  use_model_conversion = False
  epochs = 30
  use_loaded = True
  
  l = Logger(lib_name="ALNT",config_file=cfg1)
  l.SupressTFWarn()
  
  eng = ALLANTaggerEngine(log=l,)
  
  eng.initialize()
  

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
    enc_input = data[1]
    return {'tags' : res1, 'input_document':enc_input, 'id_topic_document': -1}
  
  
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
    
