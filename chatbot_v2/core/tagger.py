import argparse
import numpy as np
from functools import partial
from collections import Counter

from libraries import Logger
from libraries.training import GridSearcher
from libraries.nlp import RomBERT
import tagger.brain.utils as utils
from tagger.brain.cv_simple_model_generator import get_model

def train_generator(emb):
  
  while True:
    for step in range(n_batches):
      start = step * batch_size
      end = (step + 1) * batch_size
      if emb == 'direct':
        X = X_train_emb[start:end]
      elif emb == 'no':
        X = X_train_smp[start:end]
      elif emb == 'bert':
        X = X_train_bert[start:end]
      y = y_train[start:end] 
      yield X, y
    #endfor
  #endwhile
#end train_generator

def get_train_params_callback(**params):
  gen = train_generator(emb=params['emb'])
  if params['emb'] == 'direct':    
    X_dev = X_dev_emb
    X_train_sample = X_train_emb_sample
  elif params['emb'] == 'no':
    X_dev = X_dev_smp
    X_train_sample = X_train_smp_sample
  elif params['emb'] == 'bert':
    X_dev = X_dev_bert
    X_train_sample = X_train_bert_sample
  
  model = get_model(input_shape=X_dev.shape[1:],
                    n_classes=y_train.max() + 1, 
                    embeddings=np_embeds,
                    log=l,
                    **params
                    )

  evaluate_callbacks = [partial(evaluate_callback, X=X_train_sample, y=y_train_sample, ds='train'),
                        partial(evaluate_callback, X=X_dev, y=y_dev, ds='dev')]

  return {'model': model,
          'train_gen': gen,
          'steps_per_epoch': n_batches,
          'evaluate_callbacks': evaluate_callbacks,
          'no_model_prefix' : False}
  

def _get_shape(inp):
  if type(inp) is not list:
    return inp.shape
  else:
    return [x.shape for x in inp]

def evaluate_callback(model, epoch, X, y, ds):
  
  y_hat = model.predict(X)
  
  metrics_names = model.metrics_names

  l.P("*" * 60)
  l.P("Evaluating '{}' set: X={}; y={} on {}..."
         .format(ds, _get_shape(X), _get_shape(y), metrics_names))
  metrics_values = model.evaluate(X, y, verbose=0)
  l.P(" ", show_time=True)
  
  dct_ret = dict(zip(metrics_names, metrics_values))
  dct_ret_modif = {ds + '_' + k : v for k,v in dct_ret.items()}  
  l.P(str(dct_ret_modif))
  
  y_hat_argmax = y_hat.argmax(axis=1)
  y_true = y.squeeze()
  
  match_indexes = np.where(y_hat_argmax == y_true)[0]

  counter_labels = Counter(y_true)
  counter_labels_match = {}
  dct_labels_to_pos = {lbl:[] for lbl in counter_labels}
  
  for i,lbl in enumerate(y_true):
    dct_labels_to_pos[lbl].append(i)
  
  for k,v in dct_labels_to_pos.items():
    counter_labels_match[k] = len(set(v) & set(match_indexes))
  
  nr_print_columns = 4
  nr_print_chars = 12
  nr_labels_per_column = int(np.ceil(len(counter_labels_match) / nr_print_columns))
  print_columns = [[] for _ in range(nr_print_columns)]
  
  labels = np.array(list(counter_labels_match.keys()))
  matches = np.array(list(counter_labels_match.values()))
  pos_sort = np.argsort(matches)
  
  crt_column = 0
  for i,lbl in enumerate(labels[pos_sort]):
    str_lbl = dct_idx2label[lbl]
    nr_lbl_true = len(dct_labels_to_pos[lbl])
    nr_lbl_true_positive = counter_labels_match[lbl]
    if i // nr_labels_per_column != crt_column:
      crt_column += 1
    
    
    print_columns[crt_column].append('[{:>12}]:{}/{}'
                                     .format(str_lbl[:nr_print_chars],
                                             nr_lbl_true_positive,
                                             nr_lbl_true))
  
  l.P("Nr matches per label in '{}' set:".format(ds))
  for i in range(nr_labels_per_column):
    str_line = ''
    for j in range(nr_print_columns):
      if i >= len(print_columns[j]):
        continue
      
      str_line += print_columns[j][i] + '    '
    l.P(str_line, noprefix=True)

  l.P("*" * 60)
  return dct_ret_modif  



if __name__ == '__main__':
  VER = '02'
  base_name = 'AChT' + VER # ALLAN Chatbot Tagger
  max_size = 20 # max observation size
  batch_size = 32
  N_EPOCHS = 50
  nr_grid_iters = 50
  
  parser = argparse.ArgumentParser()
  parser.add_argument("-b", "--base_folder", help="Base folder for storage",
                      type=str, default='.')
  parser.add_argument("-a", "--app_folder", help="App folder for storage",
                      type=str, default='chatbot_v2')
  parser.add_argument("-e", "--emb_model", help="The name of embeddings model",
                      type=str, default='20200622_w2v_ep25_transfered_20200120')
  
  args = parser.parse_args()
  base_folder = args.base_folder
  app_folder = args.app_folder
  emb_model = args.emb_model
  assert emb_model is not None
  l = Logger(lib_name='ALLAN_T', base_folder=base_folder, app_folder=app_folder,
             max_lines=15000)
  
  tokens_config = {"<PAD>" : 0, "<UNK>" : 1, "<SOS>" : 2, "<EOS>" : 3}
  training_config = l.config_data['TRAINING']
  tokens = []
  for k,v in tokens_config.items():
    tokens.append(k)
  
  # train / dev data
  all_train_texts, all_train_labels, all_dev_texts, all_dev_labels = utils.load_data(l, training_config)
  all_train_labels = Logger.flatten_2d_list(all_train_labels)
  assert len(all_train_texts) == len(all_train_labels)
  assert len(all_dev_texts) == len(all_dev_labels)
  n_obs = len(all_train_labels)
  
  # load vocab
  vocab = l.load_pickle_from_models(emb_model + '.index2word.pickle')
  vocab = tokens + vocab
  dct_vocab = {w:i for i,w in enumerate(vocab)}
  
  # load embeddings
  np_embeds = np.load(l.get_model_file(emb_model + '.wv.vectors.npy'))
  x = np.random.uniform(low=-1,high=1, size=(len(tokens), np_embeds.shape[1]))
  x[tokens_config['<PAD>']] *= 0
  # withou the float32 conversion the tf saver crashes
  np_embeds = np.concatenate((x,np_embeds),axis=0).astype(np.float32)
  
  # compute labels dict
  dct_label2idx = {lbl:i for i,lbl in enumerate(list(set(all_train_labels)))}
  dct_idx2label = {v:k for k,v in dct_label2idx.items()}
  new_dev_labels = utils.check_labels_set(all_dev_labels, dct_label2idx, exclude=True)
  new_dev_labels = Logger.flatten_2d_list(new_dev_labels)
  
  y_train = np.array([dct_label2idx[lbl] for lbl in all_train_labels]).reshape(-1,1)
  y_dev = np.array([dct_label2idx[lbl] for lbl in new_dev_labels]).reshape(-1,1)
  
  dct_train_labels_to_pos = {lbl:[] for lbl in set(all_train_labels)}
  for i,lbl in enumerate(all_train_labels):
    dct_train_labels_to_pos[lbl].append(i)
  
  
  
  fct_corpus_to_batch = partial(l.corpus_to_batch,
                                tokenizer_func=utils.tokenizer,
                                dct_word2idx=dct_vocab,
                                max_size=max_size,
                                unk_word_func=None,
                                PAD_ID=tokens_config['<PAD>'],
                                UNK_ID=tokens_config['<UNK>'],
                                left_pad=False,
                                cut_left=False)
  
  
  X_train_emb = fct_corpus_to_batch(sents=all_train_texts,
                                get_embeddings=True,
                                embeddings=np_embeds)
  
  
  X_dev_emb = fct_corpus_to_batch(sents=all_dev_texts,
                            get_embeddings=True,
                            embeddings=np_embeds)
  
  
  X_train_smp = fct_corpus_to_batch(sents=all_train_texts,
                                get_embeddings=False,
                                embeddings=None)
  
  X_dev_smp = fct_corpus_to_batch(sents=all_dev_texts,
                              get_embeddings=False,
                              embeddings=None)
  
  bert = RomBERT(log=l, max_sent=max_size * 3)
  bert.text2embeds(['ana are mere'])
  X_train_bert = bert.text2embeds(all_train_texts)
  X_dev_bert = bert.text2embeds(all_dev_texts)
  
  print(X_train_emb.shape)
  print(X_dev_emb.shape)
  print(X_train_smp.shape)
  print(X_dev_smp.shape)
  print(X_train_bert.shape)
  print(X_dev_emb.shape)

  n_batches = X_train_smp.shape[0] // batch_size + 1
  
  nr_train_examples_per_label = 6
  sample_train_indexes = []
  
  for k,v in dct_train_labels_to_pos.items():
    sample_train_indexes += np.random.choice(v,
                                             min(len(v), nr_train_examples_per_label),
                                             replace=False).tolist()

  X_train_emb_sample, y_train_sample = X_train_emb[sample_train_indexes], y_train[sample_train_indexes]
  X_train_smp_sample, y_train_sample = X_train_smp[sample_train_indexes], y_train[sample_train_indexes]
  X_train_bert_sample, y_train_sample = X_train_bert[sample_train_indexes], y_train[sample_train_indexes]
  
  
  #### GRID
  main_grid = {
      "emb" : [
          'direct',
          #'no',
          #'bert',
          ],
          
      "bn" : [
          True,
          False
          ],
      
      "cols" : [
          [(1, 128), (2, 128), (3,256),(5, 256), (7, 256)],
          [(1, 32), (2, 32), (5, 32), (7, 32)],
          [(1, 64), (3, 64), (7, 64)],
          ],
          
      "ph2": [
           3,
           # 5,
           ],
      
      "fcs" : [
          [(128,True)],
          [(128,False)],
          [(256,True)],
          [(256,False)],
          [],
          ],
          
      "drp" : [
          0.3,
          0.7,
          ],
          
      "pool": [
          "avg",
          "max",
          "both",
          ]
  
      }
  

  gs = GridSearcher(grid_params=main_grid,
                    get_train_params_callback=get_train_params_callback,
                    epochs=N_EPOCHS,
                    key='dev_acc',
                    key_mode='max',
                    delete_if_key_worse=0.7,
                    stop_at_fails=20,
                    threshold_progress=0,
                    max_patience=7,
                    max_cooldown=2,
                    lr_decay=0.5,
                    return_history=False,
                    base_name=base_name,
                    log=l)
  
  gs.run(max_iters=nr_grid_iters)
  
  l.save_data_json({"ALL_LABELS" : dct_label2idx},
                   l.file_prefix + '_dct_label2idx.json')
  
  

