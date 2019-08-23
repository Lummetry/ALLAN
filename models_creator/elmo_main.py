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
              data_file_name='rowiki_dialogues_merged_v2',
              word2idx_file='rowiki_dialogues_merged_v2_wordindex_df.csv',
              max_word_length=26)

  elmo._train(epochs=10, batch_size=4)
