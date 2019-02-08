from import_utils import LoadLogger
from hierarchical_wrapper_multioutput import HierarchicalNet
import pickle
import numpy as np
#from tqdm import tqdm

from Runner import ChatBot

def compare_models(model1, model2):
  nr_layers = len(model1.layers)
  assert nr_layers == len(model2.layers)

  for i in range(nr_layers):
    l1 = model1.layers[i]
    l2 = model2.layers[i]
    
    w1 = l1.get_weights()
    w2 = l2.get_weights()
    
    nr_weights = len(w1)
    assert nr_weights == len(w2)
    
    same_layer = True
    
    for j in range(nr_weights):
      same_layer = same_layer and np.array_equal(w1[j], w2[j])

      if same_layer is not True:
        print("Layer #{},{} not same".format(i,j))
  return
  


def translate_generator_sentences(invDictionary, batch, max_words, invLabelDictionary=None):
  in_batch  = batch[0]
  out_batch = batch[1]
  in_label_batch = None
  if len(batch) == 3:
    in_label_batch = batch[2]
    assert invLabelDictionary is not None

  # print input
  for idx, sentence in enumerate(in_batch):
      new_sentence = [invDictionary[word] for i,word in enumerate(sentence) if (i < max_words) and (invDictionary[word] != '<PAD>')]
      
      if len(batch) == 3:
        label = invLabelDictionary[in_label_batch[idx]]
        print(str(new_sentence) + ' -> ' + label)
      else:
        print(str(new_sentence))

  # print output
  new_sentence = [invDictionary[word] for word in out_batch]
  print(new_sentence)


def Generator(batches, loop_forever=False):
  import random
  while True:
    random.shuffle(batches)
    for batch in batches:
      X1 = batch[0]
      X2 = batch[1][:-1]
      y1 = batch[1][1:]
      
      X1 = np.expand_dims(X1, axis=0)
      X2 = np.expand_dims(X2, axis=0)
      y1 = y1.reshape(1,-1,1)
      
      if len(batch) == 3:
        label = np.array([batch[2][-1]] * X2.shape[-1])
        label = np.expand_dims(label, axis=0)
        y2 = np.array(batch[2]).reshape(1,-1,1)
        yield([X1, X2, label], [y1, y2])
      else:
        yield([X1,X2],y1)
    
    if not loop_forever: break


if __name__ == '__main__':
  logger = LoadLogger(lib_name='DEMOBIG', config_file='config_hpc.txt', use_tf_keras=True)
  logger2= LoadLogger(lib_name='RUN', config_file='config_runner.txt', use_tf_keras=True)

  CONFIG_DATA = logger.config_data

  max_words = CONFIG_DATA['MAX_WORDS']
  fn_words_dictionary = logger.GetDataFile(CONFIG_DATA['WORDS_DICTIONARY'])
  fn_inv_words_dictionary = logger.GetDataFile(CONFIG_DATA['INV_WORDS_DICTIONARY'])
  fn_chars_dictionary = logger.GetDataFile(CONFIG_DATA['CHARS_DICTIONARY'])
  fn_inv_chars_dictionary = logger.GetDataFile(CONFIG_DATA['INV_CHARS_DICTIONARY'])
  fn_labels_dictionary = logger.GetDataFile(CONFIG_DATA['LABELS_DICTIONARY'])
  fn_inv_labels_dictionary = logger.GetDataFile(CONFIG_DATA['INV_LABELS_DICTIONARY'])
  fn_batches_good = logger.GetDataFile(CONFIG_DATA['BATCHES'])
  
  with open(fn_words_dictionary, 'rb') as handle:
    words_dictionary = pickle.load(handle)
  with open(fn_inv_words_dictionary, 'rb') as handle:
    inv_words_dictionary = pickle.load(handle)
  with open(fn_chars_dictionary, 'rb') as handle:
    chars_dictionary = pickle.load(handle)
  with open(fn_inv_chars_dictionary, 'rb') as handle:
    inv_chars_dictionary = pickle.load(handle)
  with open(fn_labels_dictionary, 'rb') as handle:
    labels_dictionary = pickle.load(handle)
  with open(fn_inv_labels_dictionary, 'rb') as handle:
    inv_labels_dictionary = pickle.load(handle)
  with open(fn_batches_good, 'rb') as handle:
    batches_good = pickle.load(handle)


  steps_per_epoch_good = len(batches_good)
  TRAIN_GENERATOR_GOOD = Generator(batches_good, loop_forever=True)


  from doc_utils import DocUtils
  d = DocUtils(logger.GetDataFile('demo_20190130/index2word_final_ep60.pickle'),
               max_nr_words=CONFIG_DATA['MAX_WORDS'], max_nr_chars=CONFIG_DATA['MAX_CHARACTERS'])
  d.CreateLabelsVocab(fn=logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190204_Production_selection_v0_1/Labels.txt')

  hnet = HierarchicalNet(logger)

  hnet.DefineTrainableModel()
#  hnet.CreatePredictionModels()

#  hnet.LoadModelWeightsAndConfig('20190130_h_enc_dec_lrdec_forcetraining_epoch200_loss0.02')
#  hnet.Fit(generator=TRAIN_GENERATOR_GOOD, nr_epochs=200,
#           steps_per_epoch=steps_per_epoch_good, save_period=50)



#  conversation = [
#      'Bine ai venit! Numele meu este Oana. Cu ce te pot ajuta?',
#      'Sarut mana doamna Oana.',
#      'Buna, <NAME>! Cum te simti astazi?',
#      'Imi este teama sa nu am o boala de piele.'
#  ]

#  hnet._step_by_step_prediction(d, conversation, method='argmax')
#  hnet._step_by_step_prediction(c, conversation, method='sampling')


#  c = ChatBot(logger2)
#  c.LoadModels()
#  c._step_by_step_prediction(d, conversation, method='argmax')