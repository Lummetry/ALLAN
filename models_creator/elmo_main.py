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
              word2idx_file='rowiki_dialogues_merged_v2_word2idx.pickle',
              max_word_length=26)
  

  elmo.corpus_tokenization()
#  for i in elmo.word2idx.keys():
#    print(i.decode("iso8859_2").encode('latin-1'))
  elmo.create_word2idx_map()

#  elmo.token_sanity_check()
  
#  elmo_model = elmo.build_model()
#          model = model.fit(x=self.training_corpus_c, y=self.training_corpus_w, batch_size=32, epochs=10, verbose=1)
  a = 0