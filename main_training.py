from import_utils import LoadLogger
from hierarchical_wrapper_multioutput import HierarchicalNet
import numpy as np
import random
from doc_utils import DocUtils

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
  logger = LoadLogger(lib_name='CHATTRAIN', config_file='config_hpc.txt', use_tf_keras=True)
  d = DocUtils(logger, logger.GetDataFile('demo_20190130/index2word_final_ep60.pickle'))
  
  d.CreateLabelsVocab(fn=logger.GetDropboxDrive() +\
                      '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190204_Production_selection_v0_1/Newlables.txt')
  d.GenerateLabels(logger.GetDropboxDrive() +\
                   '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190204_Production_selection_v0_1/labels')

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

  ########## SANITY CHECK ###############
  num_sanity_checks = 5
  logger.P("Sanity check for {} aleator training examples ...".format(num_sanity_checks))
  
  sanity_check_examples = random.sample(batches_train, num_sanity_checks)
  
  for i,exp in enumerate(sanity_check_examples):
    logger.P("Example {}/{}".format(i+1, num_sanity_checks))
    d.translate_generator_sentences(exp, translate_out_batch=True)
    logger.P("\n", noprefix=True)
  #######################################
  
  steps_per_epoch = len(batches_train)
  TRAIN_GENERATOR = TrainGenerator(batches_train, loop_forever=True, use_bot_intent=False)

  hnet = HierarchicalNet(logger, d)

  hnet.DefineTrainableModel()
  hnet.CreatePredictionModels()

  hnet.Fit(generator=TRAIN_GENERATOR, nr_epochs=200,
           steps_per_epoch=steps_per_epoch, save_period=50)