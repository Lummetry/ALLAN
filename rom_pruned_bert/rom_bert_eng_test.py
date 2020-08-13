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

from libraries import Logger
import argparse
from libraries.nlp import RomBERT

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-b", "--base_folder", help="Base folder for storage",
                      type=str, default='dropbox')
  parser.add_argument("-a", "--app_folder", help="App folder for storage",
                      type=str, default='_allan_data/_rowiki_dump')
  parser.add_argument("-v", "--vocab_size", type=int, default=100_000)
  parser.add_argument("-f", "--min_freq", type=int, default=2)
  parser.add_argument("-m", "--model", type=str, default='RoBERTa')
  
  args = parser.parse_args()
  base_folder = args.base_folder
  app_folder = args.app_folder
  vocab_size = args.vocab_size
  min_freq = args.min_freq
  model = args.model
  
  log = Logger(lib_name='LM_BERT', base_folder=base_folder, app_folder=app_folder, TF_KERAS=False)
  
  eng = RomBERT(log=log)
  res = eng.text2tokens('wlifhjejeg erg erklher ergher ekgherv klhrtn')
  print(res)
  