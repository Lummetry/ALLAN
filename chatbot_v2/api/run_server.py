from libraries import Logger
try:
  from libraries import SimpleFlaskModelServer
except:
  from model_server import SimpleFlaskModelServer
from libraries import LummetryObject

import argparse
import random
import numpy as np
from functools import partial
import re
import os

from chatbot_v2.core import bot_replicas_buckets as bot

JOB = 'JOB'
INTRO = 'INTRO'
CONVERSATION = 'CONVERSATION'
CONVERSATION_ID = 'CONVERSATION_ID'
NEXT_UTTERANCE = 'NEXT_UTTERANCE'
USER_LABEL = 'USER_LABEL'
HASHTAGS = 'HASHTAGS'
JOB_START_CONVERSATION = 'START_CONVERSATION'
JOB_INFER_CHATBOT = 'INFER_CHATBOT'
JOB_GET_HASHTAGS = 'GET_HASHTAGS'


# custom_objects: dict with 'custom_name': custom_function
def load_keras_model(model_name, custom_objects=None, DEBUG=True,
                     force_compile=True, full_path=False):
  """
  Wrapper of `tf.keras.models.load_model`

  Parameters
  ----------

  model_name : str
    The name of the model that should be loaded.

  custom_objects : dict, optional
    Custom objects that should be loaded (besides the standard ones -
    from CUSTOM_LAYERS).
    The default is None.

  DEBUG : boolean, optional
    Specifies whether the logging is enabled
    The default is True.

  force_compile : boolean, optional
    `compile` param passed to tf.keras.models.load_model
    The default is True.

  full_path : boolean, optional
    Specifies whether `model_name` is a full path or should be loaded
    from `_models` folder.
    The default is False.
  """

  try:
    from custom_layers import CUSTOM_LAYERS
  except:
    from libraries.custom_layers import CUSTOM_LAYERS

  if custom_objects is None:
    custom_objects = {}
  for k, v in CUSTOM_LAYERS.items():
    custom_objects[k] = v

  if model_name[-3:] != '.h5':
    model_name += '.h5'
  if DEBUG: log.verbose_log("  Trying to load {}...".format(model_name))

  if not full_path:
    model_full_path = os.path.join(log.get_models_folder(), model_name)
  else:
    model_full_path = model_name

  if os.path.isfile(model_full_path):
    from tensorflow.keras.models import load_model

    if DEBUG: log.verbose_log("  Loading [...{}]".format(model_full_path[-40:]))
    model = load_model(model_full_path, custom_objects=custom_objects, compile=force_compile)
    if DEBUG: log.verbose_log("  Done loading [...{}]".format(model_full_path[-40:]), show_time=True)
  else:
    log.verbose_log("  File {} not found.".format(model_full_path))
    model = None
  return model


def prepare_for_tokenization(text, remove_punctuation=True):
  text = text.lower()
  text = re.sub(r'([ \w]*)([!?„”"–,\'\.\(\)\[\]\{\}\:\;\/\\])([ \w]*)', r'\1 \2 \3', text)
  
  specials = ['\n', '\t', '\r', '\xa0', '\u2028', '\x1e']
  for s in specials:
    text = text.replace(s, ' ')
  
  if remove_punctuation:
    punctuations = '.,?!(){}/\"`~'
    for p in punctuations:
      text = text.replace(p, ' ')
  return text


def tokenizer(sentence, dct_vocab, unk_func=None):
  sentence = prepare_for_tokenization(text=sentence,
                                      remove_punctuation=True)
  
  tokens = list(filter(lambda x: x != '', sentence.split(' ')))
  ids = list(map(lambda x: dct_vocab.get(x, unk_func(x)), tokens))
  return ids, tokens


def _text_to_observation(sent, tokenizer_func, max_size, dct_word2idx,
                         unk_word_func=None,
                         get_embeddings=True, embeddings=None,
                         PAD_ID=0, UNK_ID=1, left_pad=False, cut_left=False):
  if type(sent) != str:
    log.raise_error("sent must be a string")
  if unk_word_func is None:
    unk_word_func = lambda x: UNK_ID
  ids, tokens = tokenizer_func(sent, dct_word2idx, unk_word_func)
  ids = ids[-max_size:] if cut_left else ids[:max_size]
  n_unks = ids.count(UNK_ID)

  all_unk_indices = [i for i, x in enumerate(ids) if x == UNK_ID]
  all_unk_words = set([tokens[i] for i in all_unk_indices])

  if len(ids) < max_size:
    if left_pad:
      ids = [PAD_ID] * (max_size - len(ids)) + ids
    else:
      ids = ids + [PAD_ID] * (max_size - len(ids))
  if get_embeddings:
    np_output = np.array([embeddings[i] for i in ids])
  else:
    np_output = np.array(ids)
  return np_output, n_unks, list(all_unk_words)

def corpus_to_batch(sents, tokenizer_func, max_size, dct_word2idx,
                    get_embeddings=True, embeddings=None,
                    unk_word_func=None,
                    PAD_ID=0, UNK_ID=1, left_pad=False, cut_left=False,
                    verbose=True):
  """
  sents: list of sentence
  dct_word2idx : word to word-id dict
  tokenizer_func: function with signature (sentence, dict, unk_word_tokenizer_func)
  max_size : max obs size
  embeddinds :  matrix
  get_embeddings: return embeds no ids
  PAD_ID : pad id
  UNK_ID : unknown word id
  left_pad : pad to the left
  cur_left : cut to the left
  """
  if type(sents) != list or type(sents[0]) != str:
    log.raise_error("sents must be a list of strings")

  result = [_text_to_observation(sent=x,
                                 tokenizer_func=tokenizer_func,
                                 max_size=max_size,
                                 dct_word2idx=dct_word2idx,
                                 unk_word_func=unk_word_func,
                                 get_embeddings=get_embeddings,
                                 embeddings=embeddings,
                                 PAD_ID=PAD_ID,
                                 UNK_ID=UNK_ID,
                                 left_pad=left_pad,
                                 cut_left=cut_left)
            for x in sents
            ]
  output, n_unks, all_unk_words = list(zip(*result))
  np_batch = np.array(list(output))
  if verbose:
    log.P("Tokenized {} sentences to {}. Found {} unknown words: {}".format(
      len(sents), np_batch.shape, sum(n_unks), log.flatten_2d_list(list(all_unk_words))))
  return np_batch



class Inference(LummetryObject):
  
  def __init__(self, model, dct_config_labels, thr_valid_user_label=0.1, **kwargs):
    super().__init__(**kwargs)
    self.model = model
    self.dct_config_labels = dct_config_labels
    self.dct_label2idx = self.dct_config_labels['ALL_LABELS']
    self.dct_idx2label = {v:k for k,v in self.dct_label2idx.items()}
    self.conversation_history = {}
    self.negate_label = None
    
    self._labels_descriptions = self.log.load_data_json('descriptions.txt')
    self.negate_label = self.dct_config_labels['NEGATE_LABEL']
    self.hashtags = self._compute_hashtags()
    self.thr_valid_user_label = thr_valid_user_label
    self.bot_buckets = bot.create_buckets(log=self.log,
                                          fn_buckets='buckets.json')
    self.constraint_labels = self._get_constraint_labels()
    self._check_buckets()
    return
  
  def _check_buckets(self):
    
    labels_defined_in_buckets = []
    for bucket_name in self.bot_buckets:
      labels_defined_in_buckets += self._get_expected_labels(bucket_name)
    
    labels_defined_in_buckets = set(labels_defined_in_buckets)
    nr_labels = len(labels_defined_in_buckets)
    
    all_labels_and_hashtags = list(self.dct_label2idx.keys()) + self.dct_config_labels['OTHER_HASHTAGS']
    
    self.P("Found {} labels defined in buckets".format(nr_labels))
    
    if nr_labels != len(all_labels_and_hashtags):
      self.P("  WARNING! Nr labels mismatch")
    
    for l in labels_defined_in_buckets:
      if l not in all_labels_and_hashtags:
        self.P("WARNING! Label '{}' not in all labels and hashtags list".format(l))
    
    for l in self.constraint_labels:
      if l not in all_labels_and_hashtags:
        self.P("WARNING! Constraint label '{}' not in all labels and hashtags list".format(l))
    
    for l in all_labels_and_hashtags:
      if l not in labels_defined_in_buckets:
        self.P("WARNING! Label '{}' not in expected_labels".format(l))
    
    return
    
  
  def _get_bucket_given_constraint(self, constraint_label):
    for k,v in self.bot_buckets.items():
      if 'constraint_label_before' in v:
        if v['constraint_label_before'] == constraint_label:
          return k
    return
  
  def _get_constraint_labels(self):
    constraint_labels = []
    for k,v in self.bot_buckets.items():
      if 'constraint_label_before' in v:
        constraint_labels.append(v['constraint_label_before'])
    
    return constraint_labels
  
  def _compute_hashtags(self):
    hashtags = (set(list(self.dct_label2idx.keys())) | \
                set(self.dct_config_labels['OTHER_HASHTAGS']))
    hashtags = list(hashtags - set(self.dct_config_labels['NO_HASHTAGS']))
    
    self.log.print_on_columns(*hashtags, nr_print_columns=4, header='Defined hashtags:')
    
    dct_hashtags = {}
    
    for h in hashtags:
      if h not in self._labels_descriptions:
        self.P("WARNING! Hashtag '{}' has not description".format(h))
        continue
      
      dct_hashtags[h] = self._labels_descriptions[h]
    #endfor
    
    return dct_hashtags
    
  
  def predict(self, dct_params):    
    possible_jobs = [JOB_START_CONVERSATION, JOB_INFER_CHATBOT, JOB_GET_HASHTAGS]
    
    if JOB not in dct_params:
      raise ValueError('{} key should be specified in the input JSON. Possible values: {}'
                       .format(JOB, possible_jobs))
    
    job = dct_params[JOB]
    
    if job not in possible_jobs:
      raise ValueError('Value {} for key {} should be one of these: {}'
                       .format(job, JOB, possible_jobs))
    
    
    
    if job == JOB_START_CONVERSATION:
      return self._start_conversation()
    
    if job == JOB_INFER_CHATBOT:
      if CONVERSATION not in dct_params:
        raise ValueError("The input JSON must contain the conversation until this point as value for key {}"
                         .format(CONVERSATION))
      if CONVERSATION_ID not in dct_params:
        raise ValueError("The input JSON must contain the conversation id as value for key {}"
                         .format(CONVERSATION_ID))
      return self._infer_chatbot(conversation=dct_params[CONVERSATION],
                                 conversation_id=dct_params[CONVERSATION_ID])
    
    if job == JOB_GET_HASHTAGS:
      return self._get_hashtags()
    
    return

  def _get_hashtags(self):
    return {HASHTAGS: self.hashtags}
    
  
  def _start_conversation(self):
    _conv_id = len(self.conversation_history)
    self.conversation_history[_conv_id] = {'bucket_names': ['hello'],
                                           'tip_imobil'  : None,
                                           'user_labels' : []}
    intro = self._get_replica(bucket='hello', user_labels=[])
    return {INTRO: intro, CONVERSATION_ID: _conv_id}
  
  @staticmethod
  def _find_next_bucket(prev_bucket_names, tip_imobil):
    dct_conversation_layers = bot.dct_conversation_layers_spatii
    conversation_layers = bot.conversation_layers_spatii
    
    _tmp_conversation_layers = [[] for _ in range(1+max(dct_conversation_layers.values()))]
    
    for bucket_name in prev_bucket_names:
      if bucket_name in dct_conversation_layers:
        layer_id = dct_conversation_layers[bucket_name]
        _tmp_conversation_layers[layer_id].append(bucket_name)
    
    for i in range(len(_tmp_conversation_layers)):
      tmp_layer = _tmp_conversation_layers[i]
      layer = conversation_layers[i]
      
      if set(tmp_layer) < set(layer):
        return random.choice(list(set(layer ) - set(tmp_layer)))
    #endfor
    
    return
  
  def _intersect_user_labels_with_hashtags(self, user_labels):
    intersect = []
    
    for l in user_labels:
      if l in self.hashtags:
        intersect.append(l)
        
    return intersect
  
  def _get_replica(self, bucket, user_labels, tip_imobil=None):
    replica = random.choice(self.bot_buckets[bucket]['replica'])
    replica_intro = ''
    
    if bucket in bot.insert_summarization_before:
      summarization = random.choice(self.bot_buckets['sumarizare']['replica'])
      summarization = summarization.replace(bot.sumarizare,
                                            ', '.join(self._intersect_user_labels_with_hashtags(user_labels)))
      replica_intro = summarization + '\n\n'
    
    if type(replica) is str:
      return replica_intro + replica
    
    if type(replica) is list:
      assert len(replica) == 2
      return replica_intro + replica[0].replace(bot.tip_imobil,
                                                bot.dct_tip_imobil[tip_imobil][replica[1]])
  
  def _infer_user_label(self, utterance):
    if model is None:
      return utterance

    np_inp = fct_corpus_to_batch(sents=[utterance],
                                 get_embeddings=True,
                                 embeddings=np_embeds)
    
    preds = self.model.predict(np_inp)
    
    preds = preds.ravel()
    sorted_idx = np.argsort(preds)[::-1]
    preds = preds[sorted_idx]
    
    if preds[0] < self.thr_valid_user_label:
      return 'None'
    
    user_label = self.dct_idx2label[sorted_idx[0]]
    
    if user_label in self.dct_config_labels['LABEL_TO_CERTAIN_HASHTAG']:
      return self.dct_config_labels['LABEL_TO_CERTAIN_HASHTAG'][user_label]
    
    return user_label
  
  def _append_to_bucket_names(self, conversation_id, bucket_name):
    self.conversation_history[conversation_id]['bucket_names'].append(bucket_name)
    return
  
  def _get_tip_imobil(self, conversation_id):
    return self.conversation_history[conversation_id]['tip_imobil']
  
  def _set_tip_imobil(self, conversation_id, tip_imobil):
    self.conversation_history[conversation_id]['tip_imobil'] = tip_imobil
    return
  
  def _get_user_labels(self, conversation_id):
    return self.conversation_history[conversation_id]['user_labels']
  
  def _append_to_user_labels(self, conversation_id, user_label):
    self.conversation_history[conversation_id]['user_labels'].append(user_label)
    return
  
  def _get_expected_labels(self, bucket):
    if 'expected_label_after' in self.bot_buckets[bucket]:  
      return self.bot_buckets[bucket]['expected_label_after']
    return []
  
  @staticmethod
  def _is_in_loop(bucket_name):
    return bucket_name in list(bot.loop_buckets.keys())
  
  @staticmethod
  def _is_imobil(user_label):
    return user_label in bot.locuinte + bot.spatii
  
  def _infer_chatbot(self, conversation, conversation_id):
    bucket_names = self.conversation_history[conversation_id]['bucket_names']
    last_bot_replica_bucket = bucket_names[-1]
    last_user_replica = conversation[-1]
    user_label = self._infer_user_label(last_user_replica)
    expected_labels = self._get_expected_labels(last_bot_replica_bucket)
    
    dct_ret = {NEXT_UTTERANCE: "", USER_LABEL: user_label}
    crt_tip_imobil = self._get_tip_imobil(conversation_id)

    if self._is_imobil(user_label) and crt_tip_imobil is None:
      self._set_tip_imobil(conversation_id, user_label)
      crt_tip_imobil = self._get_tip_imobil(conversation_id)
    #endif
    
    if user_label not in expected_labels:
      dct_ret[NEXT_UTTERANCE] = 'Te rog reformuleaza raspunsul la ultima intrebare.'
      return dct_ret
    #endif
    
    self._append_to_user_labels(conversation_id, user_label)
    crt_user_labels = self._get_user_labels(conversation_id)
    
    if user_label in self.constraint_labels:
      constraint_bucket = self._get_bucket_given_constraint(user_label)
      dct_ret[NEXT_UTTERANCE] = self._get_replica(bucket=constraint_bucket,
                                                  user_labels=crt_user_labels,
                                                  tip_imobil=crt_tip_imobil)
      self._append_to_bucket_names(conversation_id, constraint_bucket)
      return dct_ret
    #endif
    
    if self._is_in_loop(last_bot_replica_bucket) and user_label != self.negate_label:
      next_bucket = bot.loop_buckets[last_bot_replica_bucket]
      dct_ret[NEXT_UTTERANCE] = self._get_replica(bucket=next_bucket,
                                                  user_labels=crt_user_labels,
                                                  tip_imobil=crt_tip_imobil)
      self._append_to_bucket_names(conversation_id, next_bucket)
      return dct_ret
    
    next_bucket = self._find_next_bucket(bucket_names, crt_tip_imobil)
    if next_bucket is None:
      return dct_ret
    
    dct_ret[NEXT_UTTERANCE] = self._get_replica(bucket=next_bucket,
                                                user_labels=crt_user_labels,
                                                tip_imobil=crt_tip_imobil)
    self._append_to_bucket_names(conversation_id, next_bucket)
    return dct_ret


def start_conv():
  dct = eng.predict({JOB : JOB_START_CONVERSATION})
  return dct[INTRO], dct[CONVERSATION_ID]

def infer(conv, conv_id):
  dct = eng.predict({JOB: JOB_INFER_CHATBOT,
                     CONVERSATION: conv,
                     CONVERSATION_ID: conv_id})
  
  return dct[NEXT_UTTERANCE]

if __name__ == '__main__':
  ONLINE = False
  
  parser = argparse.ArgumentParser()
  parser.add_argument("-H", "--host", help='The host of the server', type=str, default='127.0.0.1')
  parser.add_argument("-P", "--port", help='Port on which the server runs', type=int, default=5002)
  parser.add_argument("-b", "--base_folder", help="Base folder for storage",
                      type=str, default='.')
  parser.add_argument("-a", "--app_folder", help="App folder for storage",
                      type=str, default='_temprent_deploy')
  parser.add_argument("-s", "--max_size", help='Max sentence size',
                      type=int, default=20)
  parser.add_argument("-e", "--emb_model", help="The name of embeddings model",
                      type=str)
  parser.add_argument("-m", "--model_name", help="The name of the neural model",
                      type=str)
  parser.add_argument("-d", "--dct_labels", help="The file name of dct label2idx",
                      type=str)
  
  args = parser.parse_args()
  base_folder = args.base_folder
  app_folder = args.app_folder
  host = args.host
  port = args.port
  max_size = args.max_size
  emb_model = args.emb_model
  model_name = args.model_name
  fn_dct_config_labels = args.dct_labels
  print("Running API with base_folder={} & app_folder={} on {}:{} ..."
        .format(base_folder, app_folder, host, port))

  log = Logger(lib_name='CHATBOT', base_folder=base_folder, app_folder=app_folder, max_lines=2000)
  log.reset_seeds(123, packages=['np', 'rn'])
  
  if emb_model is not None:
    tokens_config = {"<PAD>" : 0, "<UNK>" : 1, "<SOS>" : 2, "<EOS>" : 3}
    tokens = []
    for k,v in tokens_config.items():
      tokens.append(k)
    
    # load vocab
    vocab = log.load_pickle_from_models(emb_model + '.index2word.pickle')
    vocab = tokens + vocab
    dct_vocab = {w:i for i,w in enumerate(vocab)}
    log.P("Loaded vocabulary: {:,} words".format(len(dct_vocab)))
    
    # load embeddings
    np_embeds = np.load(log.get_model_file(emb_model + '.wv.vectors.npy'))
    x = np.random.uniform(low=-1,high=1, size=(len(tokens), np_embeds.shape[1]))
    x[tokens_config['<PAD>']] *= 0
    # withou the float32 conversion the tf saver crashes
    np_embeds = np.concatenate((x,np_embeds),axis=0).astype(np.float32)
    log.P("Loaded np embeds: {}".format(np_embeds.shape))
  #endif emb_model
  
  # load dct_label2idx
  dct_config_labels = None
  if fn_dct_config_labels is not None:
    dct_config_labels = log.load_dict_from_data(fn_dct_config_labels)
    log.print_on_columns(*list(dct_config_labels['ALL_LABELS'].keys()),
                         nr_print_columns=4,
                         header="Loaded dct_config_labels - current labels:")
  #endif fn_dct_label2idx
  
  #load model
  model = None
  if model_name is not None:
    model = load_keras_model(model_name)
  #endif model_name
  
  fct_corpus_to_batch = partial(corpus_to_batch,
                                tokenizer_func=tokenizer,
                                dct_word2idx=dct_vocab,
                                max_size=max_size,
                                unk_word_func=None,
                                PAD_ID=tokens_config['<PAD>'],
                                UNK_ID=tokens_config['<UNK>'],
                                left_pad=False,
                                cut_left=False,
                                verbose=False)
  
  eng = Inference(model=model, dct_config_labels=dct_config_labels, log=log)
  
  ######## TEST ZONE #########
  if True:
    conversation = []
    intro, conversation_id = start_conv()
    conversation.append(intro)
    print(intro)
    
    _input = input()
    conversation.append(_input)
    
    while _input not in ['quit', 'exit']:
      next_utterance = infer(conversation, conversation_id)
      conversation.append(next_utterance)
      print(next_utterance)
      _input = input()
      conversation.append(_input)
    #endwhile
  ############################
  
  if ONLINE:
    svr = SimpleFlaskModelServer(model=eng,
                                 log=log, host=host, port=port,
                                 db_file=None, signup=False)
    
    svr.run()
  