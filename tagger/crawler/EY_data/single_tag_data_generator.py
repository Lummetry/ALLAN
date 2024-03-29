import random
import pandas as pd

from libraries.logger import Logger


if __name__ == '__main__':


  logger = Logger(lib_name='EY_DATA',
                  config_file='./tagger/crawler/EY_data/config_eydata.txt',
                  TF_KERAS=False)
  
  data_dir = logger.GetDropboxDrive() + '/' + logger.config_data['APP_FOLDER'] + '/_data/SINGLE_TAG_CORRECT/'
  
  index = 183
  with open(data_dir + 'raw_new_data.txt', 'r') as file:
    file = file.readlines()
    for line in file:
      row = line.split()
      fname = 10000 + index
      fname = str(fname)[1:]
      with open(data_dir + 'new_batch/Texts/Text_{}.txt'.format(fname), 'w') as f_doc:
        f_doc.write(' '.join(row[:-1]))
      with open(data_dir + 'new_batch/Labels/Text_{}.txt'.format(fname), 'w') as f_lbl:
        f_lbl.write(row[-1])
      print(row[:-1])
      print(row[-1])

      index += 1