import tensorflow as tf

from tqdm import tqdm
from libraries.logger import Logger
from models_creator.elmo_model import ELMo

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
  
  logger.P('Start training...')
  epochs = 5
#  elmo_model.fit_generator(elmo.data_generator(), steps_per_epoch=1000, epochs=5, verbose=1)
  for epoch in range(epochs):
    logger.P('EPOCH {}'.format(epoch))
    for i in tqdm(elmo.data_generator(batch_size=32)): 
      loss, acc = elmo_model.train_on_batch(x=i[0], y=i[1])
      logger.P('Loss: {} Accuracy: {}'.format(loss, acc), noprefix=True)