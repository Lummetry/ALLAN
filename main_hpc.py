from import_utils import LoadLogger
from hierarchical_wrapper_multioutput import HierarchicalNet
import pickle
import numpy as np
import random

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


def translate_generator_sentences(invDictionary, batch, max_words, invLabelDictionary=None,
                                  translate_out_batch=True):
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
  if translate_out_batch:
    new_sentence = [invDictionary[word] for word in out_batch]
    print(new_sentence)
  
  return


def TrainGenerator(batches, loop_forever=False, use_bot_intent=False):
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
        # in batch[2] we have all the labels. The last but one (-2) is the last intent of the user;
        # the last is the intent of the bot
        label = np.array([batch[2][-2]] * X2.shape[-1])
        label = np.expand_dims(label, axis=0)
        y2 = np.array(batch[2][:-1]).reshape(1,-1,1)

        if use_bot_intent:
          label_bot = np.array([batch[2][-1]] * X2.shape[-1])
          label_bot = np.expand_dims(label, axis=0)
          y3 = np.array(batch[2][-1]).reshape(1,1)
          yield([X1, X2, label, label_bot], [y1, y2, y3])
        #endif

        yield([X1, X2, label], [y1, y2])
      else:
        yield([X1,X2],y1)

    if not loop_forever: break


if __name__ == '__main__':
  logger = LoadLogger(lib_name='DEMOBIG', config_file='config_hpc.txt', use_tf_keras=True)
  logger2= LoadLogger(lib_name='RUN', config_file='config_runner.txt', use_tf_keras=True)

  CONFIG_DATA = logger.config_data
  max_words = CONFIG_DATA['MAX_WORDS']

  from doc_utils import DocUtils
  d = DocUtils(logger.GetDataFile('demo_20190130/index2word_final_ep60.pickle'),
               max_nr_words=CONFIG_DATA['MAX_WORDS'], max_nr_chars=CONFIG_DATA['MAX_CHARACTERS'])
  d.CreateLabelsVocab(fn=logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190204_Production_selection_v0_1/Newlables.txt')
  d.GenerateLabels(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190204_Production_selection_v0_1/labels')

  batches_train = d.GenerateBatches(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190204_Production_selection_v0_1/texts',
                                    use_characters=True, use_labels=True, eps_words=7, eps_characters=21)

  batches_val = d.GenerateValidationBatches(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190204_Production_selection_v0_1/validare/texte_validare',
                                            use_labels=True)

  batches_train_to_validate = {}
  for b in batches_train:
    if b[0].shape[0] not in batches_train_to_validate: batches_train_to_validate[b[0].shape[0]] = []
    batches_train_to_validate[b[0].shape[0]].append(b)

  for k,v in batches_train_to_validate.items():
    batches_train_to_validate[k] = random.sample(v, min(9, len(v)))

  d.SetPredictionBatches(batches_train_to_validate, batches_val)

  inv_words_dictionary = d.dict_id2word.copy()
  inv_labels_dictionary = d.dict_id2label.copy()


  steps_per_epoch = len(batches_train)
  TRAIN_GENERATOR = TrainGenerator(batches_train, loop_forever=True, use_bot_intent=False)


#  hnet = HierarchicalNet(logger, d)
#
#  hnet.DefineTrainableModel()
#  hnet.CreatePredictionModels()

#  hnet.LoadModelWeightsAndConfig('20190130_h_enc_dec_lrdec_forcetraining_epoch200_loss0.02')
#  hnet.Fit(generator=TRAIN_GENERATOR, nr_epochs=200,
#           steps_per_epoch=steps_per_epoch, save_period=50)



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