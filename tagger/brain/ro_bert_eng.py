# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 07:29:59 2020

@author: Andrei
"""


import numpy as np
import tensorflow as tf

from libraries.logger import Logger
from libraries.generic_obj import LummetryObject

from tagger.brain import utils
from tagger.brain.utils import bert_detokenizer

from transformers import TFBertModel, BertTokenizer

class RoBERT(LummetryObject):
  def __init__(self, 
               max_sent=512,
               model_folder='_allan_data/_ro_bert/20200520', 
               **kwargs):
    super().__init__(**kwargs)
    self._model_folder = self.log.get_root_subfolder(model_folder)
    self.max_sent = 512
    self._load_models()
    return

  def _load_models(self):
    self.log.start_timer('models')
    self.P("Loading tokenizer...")
    self.tokeng = BertTokenizer.from_pretrained(self._model_folder)
    self.P("Loading pretrained tf BERT...")
    self.embeng = TFBertModel.from_pretrained(self._model_folder)
    elapsed = self.log.stop_timer('models')
    self.P("Done loading models in {:.1f}s.".format(elapsed))
    return
  
  def text2embeds(self, sent):
    if type(sent) == str:
      sent = [sent]
    elif type(sent) == list:
      assert type(sent[0]) == str, "input must be `str` or `list[str]`"
    if self.max_sent:
      data = self.tokeng.batch_encode_plus(
          sent,
          add_special_tokens=True,
          return_attention_mask=True,
          pad_to_max_length=True,
          max_length=self.max_sent
          )
    else:
      data = self.tokend.batch_encode_plus(
          sent,
          add_special_tokens=True,
          return_attention_mask=True,
          )
    np_ids = np.array(data['input_ids'])
    np_mask = np.array(data['attention_mask'])
    embeds, clf = self.embeng.predict([np_ids, np_mask])
    return embeds
      
  
if __name__ == '__main__':
  l = Logger(lib_name='ALBERT', config_file='tagger/brain/configs/config_cv_test.txt')
  
  eng = RoBERT(log=l)
  
  e = eng.text2embeds('ana are mere. mara are pere')
