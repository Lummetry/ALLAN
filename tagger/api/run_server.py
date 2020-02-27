
import numpy as np

from libraries.logger import CustomLogger
from libraries.model_server.simple_model_server import SimpleFlaskModelServer
from word_universe.doc_utils import DocUtils

from functools import partial
import argparse


def tokenizer(sentence, dct_vocab, unk_func=None):
  sentence = DocUtils.prepare_for_tokenization(text=sentence,
                                               remove_punctuation=True)
  
  tokens = list(filter(lambda x: x != '', sentence.split(' ')))
  ids = list(map(lambda x: dct_vocab.get(x, unk_func(x)), tokens))
  return ids, tokens


def input_callback(data):
  if 'CV' not in data.keys():
    return None
  s = data['CV']
  return fct_corpus_to_batch(sents=[s])



def output_callback(preds):
  preds = preds.ravel()
  sorted_idx = np.argsort(preds)[::-1]
  preds = preds[sorted_idx]
  
  payload = {
      "RESULT" : None,
      "CONFIDENCE" : None,
      "RUNNER_1" : None,
      "RUNNER_2" : None
  }
  
  confidence_thresholds = [0.73, 0.47, 0.2]
  confidence = None
  if preds[0] >= confidence_thresholds[0]:
    confidence = 'HIGH'
  elif preds[0] >=  confidence_thresholds[1]:
    confidence = 'MEDIUM'
  elif preds[0] >= confidence_thresholds[2]:
    confidence = 'LOW'
  else:
    confidence = 'LOW'
  
  payload['RESULT'] = dct_idx2label[sorted_idx[0]]
  payload['CONFIDENCE'] = confidence
  payload['RUNNER_1'] = dct_idx2label[sorted_idx[1]]
  payload['RUNNER_2'] = dct_idx2label[sorted_idx[2]]
  
  return payload
  
  



if __name__ == '__main__':
  ONLINE = False
  
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--production", help="Production mode switch on(1)/off(0)",
                      type=int, default=0)

  args = parser.parse_args()
  production = args.production
  
  if production:
    print("Running API in PRODUCTION mode..")
    config_file = 'tagger/api/config_production.txt'
  else:
    print("Running API in DEVELOPMENT mode..")
    config_file = 'tagger/api/config.txt'
  
  l = CustomLogger('NLP', 'TFKeras')(lib_name="ALNT", config_file=config_file)
  tokens_config = l.config_data['TOKENS']
  tokens = []
  for k,v in tokens_config.items():
    tokens.append(k)
  
  # max observation size
  max_size = 1400
  
  # load vocab
  vocab = l.load_pickle_from_models(l.config_data['EMB_MODEL'] + '.index2word.pickle')
  vocab = tokens + vocab
  dct_vocab = {w:i for i,w in enumerate(vocab)}
  l.P("Loaded vocabulary: {:,} words".format(len(dct_vocab)))
  
  # load embeddings
  np_embeds = np.load(l.get_model_file(l.config_data['EMB_MODEL'] + '.wv.vectors.npy'))
  x = np.random.uniform(low=-1,high=1, size=(len(tokens), np_embeds.shape[1]))
  x[tokens_config['<PAD>']] *= 0
  np_embeds = np.concatenate((x,np_embeds),axis=0).astype(np.float32)
  l.P("Loaded np embeds: {}".format(np_embeds.shape))
  
  # load dct_label2idx
  dct_label2idx = l.load_dict_from_data(l.config_data['DCT_LABEL2IDX'])
  dct_idx2label = {v:k for k,v in dct_label2idx.items()}
  l.P("Loaded dct_label2idx - current labels: {}".format(list(dct_label2idx.keys())))
  
  # load model
  model = l.load_keras_model(l.config_data['MODEL'])
  l.log_keras_model(model)

  fct_corpus_to_batch = partial(l.corpus_to_batch,
                                tokenizer_func=tokenizer,
                                dct_word2idx=dct_vocab,
                                max_size=max_size,
                                unk_word_func=None,
                                PAD_ID=tokens_config['<PAD>'],
                                UNK_ID=tokens_config['<UNK>'],
                                left_pad=False,
                                cut_left=False,
                                get_embeddings=True,
                                embeddings=np_embeds)
  
  inp = input_callback({
	"CV" : "Calculez si mut (tva-ul ma refer) ca masina de cusut. Bine si mai stiu si excel"
  })
    
  preds = model.predict(inp)
  
  out = output_callback(preds)
  
  
  if ONLINE:
    simple_server = SimpleFlaskModelServer(model=model,
                                           predict_function='predict',
                                           fun_input=input_callback,
                                           fun_output=output_callback,
                                           log=l,
                                           host=l.config_data['HOST'],
                                           port=l.config_data['PORT'])
    simple_server.run()
  
  
  
  