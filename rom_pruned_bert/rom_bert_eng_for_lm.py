"""
Copyright 2019 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


* NOTICE:  All information contained herein is, and remains
* the property of Knowledge Investment Group SRL.  
* The intellectual and technical concepts contained
* herein are proprietary to Knowledge Investment Group SRL
* and may be covered by Romanian and Foreign Patents,
* patents in process, and are protected by trade secret or copyright law.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Knowledge Investment Group SRL.


@copyright: Lummetry.AI
@author: Lummetry.AI
@project: 
@description:
"""



def test_model(model, tokenizer, strings):
  fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
  )
  
  for s in strings:
    log.P("<mask> candidates for '{}':".format(s))
    ret = fill_mask(s)
    for candidate in ret:
      log.P("* {}".format(candidate), noprefix=True)
  #endfor
  return
#enddef


from libraries import Logger
import argparse
from transformers import pipeline, BertForMaskedLM, BertTokenizer

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-b", "--base_folder", help="Base folder for storage",
                      type=str, default='dropbox')
  parser.add_argument("-a", "--app_folder", help="App folder for storage",
                      type=str, default='_allan_data/_ro_bert')
  parser.add_argument("-v", "--vocab_size", type=int, default=100_000)
  parser.add_argument("-f", "--min_freq", type=int, default=2)
  parser.add_argument("-m", "--model", type=str, default='RoBERTa')
  
  DEFAULT_MODEL = '20200520'
  
  args = parser.parse_args()
  base_folder = args.base_folder
  app_folder = args.app_folder
  vocab_size = args.vocab_size
  min_freq = args.min_freq
  model = args.model
  
  log = Logger(
    lib_name='LM_BERT',
    base_folder=base_folder,
    app_folder=app_folder,
    TF_KERAS=False
  )
  
  
  tokenizer = BertTokenizer.from_pretrained(log.get_base_subfolder(DEFAULT_MODEL))
  model = BertForMaskedLM.from_pretrained(log.get_base_subfolder(DEFAULT_MODEL))
  
  list_s = ["Mi-am luat Tesla si ma dau cu ea prin <mask>.",
            "Azi mi-am luat <mask> si ma duc la cumparaturi."]
  
  test_model(
    model=model,
    tokenizer=tokenizer,
    strings=list_s)
  