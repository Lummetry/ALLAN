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
              fn_word2idx='rowiki_dialogues_merged_v2_wordindex_df.csv')
  elmo.corpus_tokenization()
  elmo.train()
  
  elmo.get_elmo('Testez modelul.')
  
  elmo.get_elmo('Incerc sa stric modelul. Trebuie testat. Neaparat.')
  
  elmo.heat_map_sentence_similarity('Numele meu este Alex, tu cum te numesti?')
  elmo.heat_map_sentence_similarity('Clinica asta este dureure. Vreau alt spital.')