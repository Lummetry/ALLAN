from models_creator.elmo_model import ELMo
from libraries.logger import Logger

#method works both char2index and index2chars, could work words to wordindex and wordindex to char 
def index_to_characters(dictionary, word_vector):
  word = []
  for i in word_vector:
    word.append(dictionary.get(i))
    
  return word


if __name__ == '__main__':
  logger = Logger(lib_name='RO-ELMo', 
                  config_file='./models_creator/config_elmo.txt',
                  TF_KERAS = True,
                  SHOW_TIME = True
                  )

  elmo = ELMo(logger,
              data_file_name='rowiki_dialogues_merged_v2',
              max_word_length=26)
  
  
  elmo.corpus_tokenization()
  elmo.build_model()
  
  a = 0