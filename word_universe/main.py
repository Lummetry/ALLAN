import os
import pickle
from itertools import chain
from collections import Counter

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
  file_prefix = log.file_prefix[:NR_AAAALLZZ]
  corpus_folder = log.config_data['CORPUS_FOLDER']
  path_corpus_folder = os.path.join(log.get_data_folder(), corpus_folder)
  assert os.path.isdir(path_corpus_folder),\
         "Corpus folder '{}' does not exist".format(path_corpus_folder)
  numbers_in_corpus_folder = sum(c.isdigit() for c in corpus_folder)
  suffix = '_corpus_merged'
  corpus_file = file_prefix + suffix
  compute_corpus_file = True

  prefix = corpus_folder[:NR_AAAALLZZ]
  if all(c.isdigit() for c in prefix):
    files = list(filter(lambda x: prefix in x and suffix in x,
                        os.listdir(log.get_data_folder())))
    if len(files) > 0:
      assert len(files) == 1
      corpus_file = files[0]
      log.P("Found already computed corpus file: {}".format(corpus_file)) 
      compute_corpus_file = False
    else:
      corpus_file = prefix + suffix
    
    file_prefix = prefix
  #endif
  
  corpus_file = os.path.join(log.get_data_folder(), corpus_file)
  
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
  model_name = file_prefix + '_w2v_ep{}'.format(n_epochs)
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
    model_name += '_transfered_{}'.format(model_transfered_name[:NR_AAAALLZZ])
    model = Word2Vec.load(transfer_model_real_path)
    model.vocabulary.min_count = 1
    
    old_vocab = set()
    old_vocab.update(model.wv.index2word)
    
    new_vocab = set()
    flattened_sentences = list(chain.from_iterable(sentences))
    new_vocab.update(flattened_sentences)
    cnt_new_vocab = Counter(flattened_sentences)

    new_words_added = new_vocab - old_vocab
    log.P("Added {} new words in vocab:".format(len(new_words_added)))
    str_format = "{:>15} : {:<4}"
    log.P(str_format.format("Word", "Count"), noprefix=True)
    dct_cnt_new_vocab = {x: cnt_new_vocab[x] for x in cnt_new_vocab if x in new_words_added}
    lst_cnt_new_vocab = list(dct_cnt_new_vocab.items())
    lst_cnt_new_vocab = sorted(lst_cnt_new_vocab, key=lambda x: x[1])[::-1]
    
    for w,c in lst_cnt_new_vocab:
      log.P(str_format.format(w[:15], c), noprefix=True)
    
    model.build_vocab(sentences, update=True)
    model.train(sentences=sentences,
                total_examples=len(sentences),
                epochs=n_epochs,
                start_alpha=0.005,
                end_alpha=0.001,
                compute_loss=True,
                callbacks=[epoch_logger])
    

  model.callbacks = []
  model.save(os.path.join(log.get_models_folder(), model_name))
  with open(os.path.join(log.get_models_folder(),
                         model_name + '.index2word.pickle'), 'wb') as handle:
    pickle.dump(model.wv.index2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
