# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 07:29:59 2020

@author: Andrei
"""


import numpy as np
import tensorflow as tf

from libraries.logger import Logger

  
if __name__ == '__main__':
  l = Logger(lib_name='ALBERT', config_file='tagger/brain/configs/config_cv_test.txt')


  from libraries.nlp import RomBERT
  
  from transformers import TFBertModel, BertTokenizer
  
  eng = RomBERT(log=l)
  
  e = eng.text2embeds(['ana are mere.',' mara are pere'])
  
  eng.D("Result : {}".format(e.shape))
  l.P("TEST", color='green')
