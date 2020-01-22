import os
import pickle

from libraries.logger import Logger
from chatbot.word_universe.create_rowiki_dump import parse_and_merge_files

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import time

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.start_time = time.time()

    def on_epoch_begin(self, model):
        self.start_time = time.time()
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        elapsed_time = time.time() - self.start_time
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print("Epoch #{} end. Loss: {} Elapsed time: {}".format(self.epoch,
                            model.get_latest_training_loss(),
                            time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        self.epoch += 1

if __name__ == '__main__':
  
  log = Logger(lib_name='ALLANVOCAB', config_file='chatbot/word_universe/config.txt',
               TF_KERAS=False)
  
  file_prefix = log.file_prefix
  corpus_folder = os.path.join(log.GetDataFolder(), log.config_data['CORPUS_FOLDER'])
  corpus_file = os.path.join(log.GetDataFolder(), file_prefix + '_rowiki')
  
  log.P("Parsing and merging files from {} ...".format(corpus_folder))
  parse_and_merge_files(corpus_folder, corpus_file)
  
  log.P("Reading sentences from {} ...".format(corpus_file))
  sentences = []
  with open(corpus_file, "rt", encoding='utf8') as f:
    for line in f.readlines():
      sentences.append(line.split())

  epoch_logger = EpochLogger()

  log.P("Modelling word2vec ...")
  n_epochs = 1
  model = Word2Vec(sentences=sentences,
                  iter=n_epochs,
                  sg=1,
                  alpha=0.005,
                  min_alpha=0.001,
                  min_count=15,
                  window=2,
                  size=128,
                  negative=25,
                  workers=12,
                  compute_loss=True,
                  callbacks=[epoch_logger])
  
  name = file_prefix + '_w2v_ep{}'.format(n_epochs)
  model.callbacks = []
  model.save(os.path.join(log.GetModelsFolder(), name))
  with open(os.path.join(log.GetModelsFolder(),
                         name + '.index2word.pickle'), 'wb') as handle:
    pickle.dump(model.wv.index2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
