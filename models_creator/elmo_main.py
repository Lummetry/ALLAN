import warnings

from libraries.logger import Logger
from models_creator.elmo_model import ELMo


#remove pesky sklearn warnings...
def warn(*args, **kwargs):
    pass
warnings.warn = warn

warnings.filterwarnings('always')

if __name__ == '__main__':
  logger = Logger(lib_name='RO-ELMo', 
                  config_file='./models_creator/config_elmo.txt',
                  TF_KERAS = True,
                  SHOW_TIME = True)
  
  logger.SetNicePrints()
  
  elmo = ELMo(logger,
              fn_data='rowiki_dialogues_merged_v2',
              fn_word2idx='rowiki_dialogues_merged_v2_wordindex_df.csv',
              max_word_length=26)

  elmo.train(epochs=10, batch_size=4)
