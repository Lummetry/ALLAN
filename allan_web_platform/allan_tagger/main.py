import sys
import time
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0,"D:\\cs\\Lummetry.AI\\ALLAN\\")

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from models_creator.doc_utils import DocUtils
from models_creator.import_utils import LoadLogger
from bokeh.models import TextAreaInput, Paragraph, Button, Div
from bokeh.models.widgets import MultiSelect



def index_to_characters(dictionary, word_vector):
  word = []
  for i in word_vector:
    word.append(dictionary.get(i))
    
  return word

def percentage_of_unique_words(text, unique_words):
  number = 0
  text_split = text.split()
  for i in text_split:
    if i in unique_words:
      number += 1 
      
  return (number/len(text_split))*100 
  
def predict_new_doc(d, text, model, unique_words):
  found_labels_and_probabilities = []
  lines = []
  
  #tokenize input
  lines.append(text)
  tokenized_text, _ = d.tokenize_single_conversation(lines)
  tokenized_text = tokenized_text[0]
  
  #compute percentage of words in user input that also appear in the tagger's training data
  percentage = percentage_of_unique_words(text, unique_words)
  
  #pad when necessary
  if len(tokenized_text) < 257:
    tokenized_text = tokenized_text + (257 - len(tokenized_text)) * [d.dict_word2id["PAD"]]
  
  #trim when necessary
  if len(tokenized_text) > 257:
    error_text = "Text is too long! Will use only first 257 words for tagging..."
    tokenized_text = tokenized_text[:257]
    p_error = Paragraph(text=error_text,
    width=200, height=100)
    curdoc().add_root(p_error)
  
  #get highest scoring label predictions
  logger.P('Tokenized input sentence [{}]'.format(text))
  inp = np.expand_dims(np.array(tokenized_text), axis=0) 
  yhat = model.predict(np.array(inp))[0]
  best = np.argsort(yhat)[::-1][:3]
  
  #log and return them
  logger.P('Highest confidence labels, and their scores are')
  for i in best:
    logger.P('Pred Category: "{}" with probability: {}'.format(d.dict_master_id2label[i], yhat[i]), noprefix=True)
    found_labels_and_probabilities.append((d.dict_master_id2label[i], yhat[i]))
  
  return found_labels_and_probabilities, percentage

#if main nu merge cu bokeh serve --show .
#if __name__ == '__main__':
  
  
#fix for: failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#end fix
 
#Load Logger
logger = LoadLogger(lib_name='DOCSUMM-WEB-APP', config_file='../../models_creator/config_tagger.txt',
                    use_tf_keras=True)

#load index2word dict
d = DocUtils(logger, logger.GetDataFile('demo_20190611/20190611_ep55_index2word.pickle'))

#load document labels
d.CreateMasterLabelsVocab(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190226_Production_selection_v0_3/master_labels')
d.GenerateMasterLabels(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190226_Production_selection_v0_3/master_labels')

# =============================================================================
# CODE FOR GENERATING 'unique_words_in_mihaiwork_productionselection_v3.pkl'  FILE!
     #load training data, to create the set of unique words
     #
     #training_corpus_w, _ = d.tokenize_conversations(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190226_Production_selection_v0_3/texts')
     #
     #unique_words = set('')
     #for text in list(training_corpus_w.values()):
     #  for sentence in text:
     #    for word in sentence:
     #      unique_words.add(word)
     #
     #print(index_to_characters(d.dict_id2word,list(unique_words)))
     #
     #import pickle as pkl
     #
     #saved_unique_words = [list(unique_words),index_to_characters(d.dict_id2word,list(unique_words))]
     #    
     #with open("unique_words_in_mihaiwork_productionselection_v3.pkl", "wb") as f:
     #  pkl.dump(saved_unique_words, f)
# =============================================================================

#load unique words
unique_words = pd.read_pickle(logger.GetDropboxDrive() + '\_allan_data\_allan_tagger\_models\\unique_words_in_mihaiwork_productionselection_v3.pkl')
unique_words = unique_words[1]

random.shuffle(unique_words)

#display unqiue words 
multi_select = MultiSelect(title="Unique words in training:", value=["foo"],
                           options=unique_words, width=350, height=500)

#load pretrained model
tagger_model = tf.keras.models.load_model(logger.GetDropboxDrive() + '\_allan_data\_allan_tagger\_models\\allan_tagger_pretrained_model.h5')

#user_input lists 
document_list = []
findings_list = []
percentage_list = []

#callback to predict on user_input 
def callback_print(args, old, new):
  if new:
    document_list.append(new)
    preds, percs = predict_new_doc(d, new, tagger_model, unique_words)
    findings_list.append(preds)
    percentage_list.append(percs)

#display % of words in user input present in training data
welcome_text = 'Buna, eu sunt ALLAN - Doc Tagger. Am fost antrenat pe 100 de conversatii, cu un total de 1602 cuvinte unice - prezente in lista de alaturi.'
p_welcome = Paragraph(text=welcome_text,
width=200, height=200)

# user interaction elements
text_input = TextAreaInput(value="", title="Introdu documentul pe care doresti sa il etichietezi...", align = "center", width = 400, height = 300)
text_input.on_change("value", callback_print)

bt = Button(label='Tag!')

#page layout
static_data = row(p_welcome, multi_select)
interactiv = widgetbox(text_input, bt)
app_layout = row(static_data, interactiv)

#button callback
#after preparing the predictions and percentage paragraphs, button clears screen and rerenders everything in place
#otherwise predictions would be stacked on top of one another, making for unreadable text

def change_click():
  #display findings 
  findings_text = "In urma procesarii documentului tau, propun urmatoarele labeluri: <br> Label 1: <b>{}</b> cu incredere de {}  <br> Label 2: <b>{}</b> cu incredere de {}  <br> Label 3: <b>{}</b> cu incredere de {}".format(findings_list[-1][0][0], str(findings_list[-1][0][1])[:5], findings_list[-1][1][0], str(findings_list[-1][1][1])[:5], findings_list[-1][2][0], str(findings_list[-1][2][1])[:5])
  p = Div(text = findings_text,
  width=400, height=100)
  
  #display percentage
  percentage_text = "In documentul introdus, <b>{} \% </b> din cuvinte au fost prezente si in datele mele de antrenare...".format(str(percentage_list[-1])[:4])
  p_percentage = Div(text=percentage_text,
  width=400, height=60)
  
  #prep page layout for render
  static_data = row(p_welcome, multi_select)
  interactiv = widgetbox(text_input, bt)
  returns = column(interactiv, p_percentage, p)
  app_layout = row(static_data, returns)

  
  #clear screen
  curdoc().clear()
  
  #render new information
  curdoc().add_root(app_layout)
  curdoc()

#button function implementation
bt.on_click(change_click)

#layout render on initial page load
curdoc().add_root(app_layout)
