from models_creator.elmo_model import ELMo
from libraries.logger import Logger

#NEED TO MENTION: dictionaru word2index are entryuri si pentru 'Ce' si pentru 'ce'  -- vrem asta? probabil-  din moment ce char level cele doua sunt clar diferite.

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


  elmo.corpus_tokenization()

  elmo.token_sanity_check()
  
  elmo_model = elmo.build_model()
#  elmo_model.fit(x=elmo.training_corpus_c, y=elmo.training_corpus_w_idx, batch_size=32, epochs=10)