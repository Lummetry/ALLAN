import random
import time
import numpy as np
import tensorflow as tf

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from models_creator.doc_utils import DocUtils
from libraries.logger import Logger
from bokeh.models import TextAreaInput, Paragraph, Button, Div
from bokeh.models.widgets import MultiSelect


def index_to_characters(dictionary, word_vector):
  word = []
  for i in word_vector:
    word.append(dictionary.get(i))
    
  return word

class Tagger_Web(object):
  def __init__(self, logger, model_name, model_input_len, word_dict_file, path_to_training_data):
    start_time = time.time()
    self.logger = logger
    self.model = self.logger.LoadKerasModel(model_name)
    self.model_input_len = model_input_len
    self.d = DocUtils(self.logger, self.logger.GetDataFile(word_dict_file))
    self.training_corpus, _ = self.d.tokenize_conversations(self.logger.GetDropboxDrive() + path_to_training_data + 'texts')
    self.d.CreateMasterLabelsVocab(logger.GetDropboxDrive() + path_to_training_data + 'master_labels')
    self.d.GenerateMasterLabels(logger.GetDropboxDrive() + path_to_training_data + 'master_labels')
    
    self.document_list = []
    self.findings_list = []
    self.percentage_list = []
    self.unknown_words_list = []
    
    load_time = time.time() - start_time
    logger.P('Took {} seconds to load data'.format(str(load_time)[:7]))
    start_time = time.time()
    #create unique words set
    self.unique_word_ids = set()
    for text in list(self.training_corpus.values()):
      for sentence in text:
        for word in sentence:
          self.unique_word_ids.add(word)
          
    self.unique_word_str = index_to_characters(self.d.dict_id2word,list(self.unique_word_ids))
    random.shuffle(self.unique_word_str)
    self.logger.P('In {} conversations used in training, found {} unique words...'.format(len(self.training_corpus),len(self.unique_word_str)))
    
    logger.P('Took another {} seconds to get unique words'.format(str(time.time() - start_time)[:7]))

  #predict on new string function
  def predict_new_doc(self, text):
    lines = text.split('\n') 
    found_labels_and_probabilities = []
    
    #tokenize input
    tokenized_text, _ = self.d.tokenize_single_conversation(lines)
    unknown_words = self.d.unknown_words_per_conv
    
    #compute percentage of words in user input that also appear in the tagger's training data
    number = 0
    total = 0
    for line_t in tokenized_text:
      total += len(line_t)
      common = [i for i in line_t if i in self.unique_word_ids and i != self.d.dict_word2id['<UNK>']]
      number += len(common)
      
    percentage = (number/total)*100 
    
    #pad when necessary
    tokenized_text = self.d.flatten2d(tokenized_text)
    self.document_list.append(tokenized_text)
    if len(tokenized_text) < self.model_input_len:
      tokenized_text = tokenized_text + (self.model_input_len - len(tokenized_text)) * [self.d.dict_word2id['<PAD>']]
    
    #trim when necessary
    if len(tokenized_text) > self.model_input_len:
      error_text = 'Text is too long! Will use only first 257 words for tagging...'
      tokenized_text = tokenized_text[:self.model_input_len]
      p_error = Paragraph(text=error_text, 
                          width=200, 
                          height=100)
      
      curdoc().add_root(p_error)
    
    #get highest scoring label predictions
    self.logger.P('Tokenized input sentence [{}]'.format(text))
    inp = np.expand_dims(np.array(tokenized_text), axis=0) 
    yhat = self.model.predict(np.array(inp))[0]
    best = np.argsort(yhat)[::-1][:3]
    
    #log and return them
    self.logger.P('Highest confidence labels, and their scores are')
    for i in best:
      self.logger.P('Pred Category: "{}" with probability: {}'.format(self.d.dict_master_id2label[i], yhat[i]), noprefix=True)
      found_labels_and_probabilities.append((self.d.dict_master_id2label[i], yhat[i]))
    
    return found_labels_and_probabilities, percentage, unknown_words
  
  #textinput callback: adding user input to list
  def callback_print(self, args, old, new):
    if new:
      preds, percs, unknown_words = app.predict_new_doc(new.rstrip())
      self.findings_list.append(preds)
      self.percentage_list.append(percs)
      self.unknown_words_list.append(unknown_words)
  
  #button callback: reload page elements with prediction   
  def change_click(self):
  #display findings
    crt_unknown_words = []
    findings_text = 'In urma procesarii documentului tau, propun urmatoarele labeluri: <br> Label 1: <b>{}</b> cu incredere de {}  <br> Label 2: <b>{}</b> cu incredere de {}  <br> Label 3: <b>{}</b> cu incredere de {}'.format(self.findings_list[-1][0][0], str(self.findings_list[-1][0][1])[:5], self.findings_list[-1][1][0], str(self.findings_list[-1][1][1])[:5], self.findings_list[-1][2][0], str(self.findings_list[-1][2][1])[:5])
    self.p = Div(text = findings_text,
                 width=400, 
                 height=100)
    
    #display percentage
    percentage_text = "In documentul introdus, <b>{} % </b> din cuvinte au fost prezente si in datele mele de antrenare...".format(str(self.percentage_list[-1])[:4])
    self.p_percentage = Div(text=percentage_text, 
                            width=400, 
                            height=60)
    
    #highlight words not in training data
    user_input = self.document_list[-1]
    crt_unknown_words = self.unknown_words_list[-1]
    highlight_text = 'Cuvintele ingrosate nu sunt prezente in datele mele de antrenare: <br> <br>'
    for i in user_input:
      s = self.d.dict_id2word.get(i)
      if s == '<UNK>':
        s = '<b>UNK</b>'
      if i in self.unique_word_ids:
        highlight_text = highlight_text + ' ' + s 
      else:
        highlight_text = highlight_text + ' <b>' + s + '</b>'
      
      highlight_text = highlight_text
      
        
#    highlight_text = self.d.organize_text(highlight_text)
    self.highlight_div = Div(text = highlight_text,
                             width = 400,
                             height = 300)
    
    
    # TODO afisare pe ecran un text de genul: Cuvinte folosite in document pe care nu le am in vocabularul meu: {}.format(crt_unknown_words)
    if len(crt_unknown_words) == 0:
      unk_text = 'Chiar daca unele cuvinte nu sunt in datele de antrenare, toate cuvintele sunt prezente in <i>vocabularul</i> meu.'
    else: 
      unk_text = 'Cuvintele din in documentul tau care nu se regasesc in <i>vocabularul</i> meu sunt urmatoarele: <br>'
      for i in crt_unknown_words:
        unk_text = unk_text + str(i) + '<br>'
    
    self.unk_div = Div(text = unk_text,
                             width = 300,
                             height = 200)
    
    
    #empty textbox
    self.text_input.value = ''
    #prep page layout for render
    self.static_data = row(self.p_welcome, self.multi_select)
    self.interactiv = widgetbox(self.text_input, self.bt)
    self.returns = column(self.interactiv, self.p_percentage, self.p)
    self.user_input_analysis = column(self.highlight_div, self.unk_div)
    self.app_layout = row(self.static_data, self.returns, self.user_input_analysis)
  
    
    #clear screen
    curdoc().clear()
    
    #render new information
    curdoc().add_root(self.app_layout)
    curdoc()
  
  def init_web_app(self):
    self.multi_select = MultiSelect(title="Unique tokens in training data:", value=['uni'], 
                               options=app.unique_word_str, 
                               width=350, 
                               height=500)
    
    welcome_text = 'Buna, eu sunt ALLAN - Doc Tagger. Am fost antrenat pe {} de conversatii, cu un total de {} cuvinte unice - prezente in lista de alaturi.'.format(len(app.training_corpus), len(app.unique_word_str))
    self.p_welcome = Paragraph(text=welcome_text, 
                               width=200, 
                               height=200)
    
    # user interaction elements
    self.text_input = TextAreaInput(value='', 
                                    title='Introdu documentul pe care doresti sa il etichietezi...', 
                                    width = 400, 
                                    height = 300)
    
    self.text_input.on_change('value', self.callback_print)
    
    self.bt = Button(label='Tag!')
    
    #page layout
    self.static_data = row(self.p_welcome, self.multi_select)
    self.interactiv = widgetbox(self.text_input, self.bt)
    self.app_layout = row(self.static_data, self.interactiv)

    #button function implementation
    self.bt.on_click(self.change_click)
    
    #layout render on initial page load
    curdoc().add_root(self.app_layout)


#if main nu merge cu bokeh serve --show .
#if __name__ == '__main__':
  
#fix for: failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#end fix
 
#Load Logger
logger = Logger(lib_name='DOCSUMM-WEB-APP', 
                config_file='.\config_tagger.txt',
                TF_KERAS=True)

app = Tagger_Web(logger,
                 model_name = 'allan_tagger_pretrained_model',
                 model_input_len=257,
                 word_dict_file = '20190611_ep55_index2word.pickle',
                 path_to_training_data = '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190226_Production_selection_v0_3/'
                 )


app.init_web_app()
