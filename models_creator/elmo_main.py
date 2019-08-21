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
  epochs = 10
  batch_size = 4
  
  elmo.build_batch_list(batch_size)
  training_steps = elmo.number_of_training_batches
  validation_steps = elmo.number_of_validation_batches
  
  elmo_model.fit_generator(elmo.train_generator(), 
                           steps_per_epoch=training_steps, 
                           epochs=epochs,
                           validation_data=elmo.validation_generator(),
                           validation_steps=validation_steps)
