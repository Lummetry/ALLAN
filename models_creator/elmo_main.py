from tqdm import tqdm
from libraries.logger import Logger
from models_creator.elmo_model import ELMo

def flatten_list(a):
  return [item for sublist in a for item in sublist]

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
#  elmo._train(epochs)
#  for epoch in range(epochs):
#    logger.P('EPOCH {}'.format(epoch))
#    for i in tqdm(elmo.data_generator(batch_size=32)): 
#      loss, acc, rec, prec = elmo_model.train_on_batch(x=i[0], y=i[1])
#      logger.P('Loss: {} Accuracy: {} Recall: {} Precision: {}'.format(loss, acc, rec, prec), noprefix=True)
  batch_size = 3 
  elmo.build_batch_list(batch_size)
  steps = elmo.number_of_batches 
  elmo_model.fit_generator(elmo.data_generator(epochs), steps_per_epoch=steps, epochs=epochs)
#  elmo_model.fit_generator(elmo.data_generator(batch_size=4), steps_per_epoch=5, epochs=10)