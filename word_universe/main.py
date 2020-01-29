import os
import pickle

from libraries.logger import Logger
from word_universe.create_rowiki_dump import parse_and_merge_files

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
  
  log = Logger(lib_name='ALLANVOCAB', config_file='word_universe/config.txt',
               TF_KERAS=False)
  
  NR_AAAALLZZ = 8
  NR_AAAALLZZ_HHMMSS = 15
  file_prefix = log.file_prefix[:NR_AAAALLZZ]
  corpus_folder = log.config_data['CORPUS_FOLDER']
  path_corpus_folder = os.path.join(log.GetDataFolder(), corpus_folder)
  assert os.path.isdir(path_corpus_folder),\
         "Corpus folder '{}' does not exist".format(path_corpus_folder)
  numbers_in_corpus_folder = sum(c.isdigit() for c in corpus_folder)
  suffix = '_corpus_merged'
  corpus_file = file_prefix + suffix
  compute_corpus_file = True
  if numbers_in_corpus_folder == NR_AAAALLZZ:
    prefix = corpus_folder[:numbers_in_corpus_folder]
    if all(c.isdigit() for c in prefix):
      files = list(filter(lambda x: prefix in x and suffix in x,
                          os.listdir(log.GetDataFolder())))
      if len(files) > 0:
        assert len(files) == 1
        corpus_file = files[0]
        log.P("Found already computed corpus file: {}".format(corpus_file)) 
        compute_corpus_file = False
      else:
        corpus_file = prefix + suffix
  
  
  corpus_file = os.path.join(log.GetDataFolder(), corpus_file)
  
  if compute_corpus_file:
    log.P("Parsing and merging files from {} ...".format(path_corpus_folder))
    parse_and_merge_files(path_corpus_folder, corpus_file)
  
  log.P("Reading sentences from {} ...".format(corpus_file))
  sentences = []
  with open(corpus_file, "rt", encoding='utf8') as f:
    for line in f.readlines():
      sentences.append(line.split())

  epoch_logger = EpochLogger()

  log.P("Modelling word2vec ...")
  n_epochs = 25
  model_name = log.file_prefix + '_w2v_ep{}'.format(n_epochs)
  transfer_model_real_path = log.config_data['TRANSFER_MODEL_REAL_PATH'].replace('\\', '/')  
  if transfer_model_real_path == "":
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
  else:
    model_transfered_name = transfer_model_real_path.split('/')[-1]
    model_name += '_transfered_{}'.format(model_transfered_name[:NR_AAAALLZZ_HHMMSS])
    model = Word2Vec.load(transfer_model_real_path)
    
  model.callbacks = []
  model.save(os.path.join(log.GetModelsFolder(), model_name))
  with open(os.path.join(log.GetModelsFolder(),
                         model_name + '.index2word.pickle'), 'wb') as handle:
    pickle.dump(model.wv.index2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
