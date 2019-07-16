import tensorflow as tf
import numpy as np
import time
import sys

sys.path.insert(0,"D:\\cs\\Lummetry.AI\\ALLAN\\")

from models_creator.import_utils import LoadLogger
from models_creator.doc_utils import DocUtils
from bokeh.layouts import widgetbox
from bokeh.models import CustomJS, TextInput, Paragraph, Button
from bokeh.plotting import output_file, show
from bokeh.io import curdoc

def predict_new_doc(d, text, model):
  
  tokenized_text = d.tokenize_single_conversation(text)
  
  tokenized_text = d.flatten2d(tokenized_text[0])
  
  found_labels_and_probabilities = []
  
  #pad when necessary
  if len(tokenized_text) < 257:
    tokenized_text = tokenized_text + (257 - len(tokenized_text)) * [d.dict_word2id["PAD"]]

  #NEED TO TRIM!!! 
  
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
  
  return found_labels_and_probabilities

#if __name__ == '__main__':
  
#Load Logger
logger = LoadLogger(lib_name='DOCSUMM-WEB-APP', config_file='../../models_creator/config_tagger.txt',
                    use_tf_keras=True)

#load index2word dict
d = DocUtils(logger, logger.GetDataFile('demo_20190611/20190611_ep55_index2word.pickle'))

#load document labels
d.CreateMasterLabelsVocab(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190226_Production_selection_v0_3/master_labels')
d.GenerateMasterLabels(logger.GetDropboxDrive() + '/_doc_ro_chatbot_data/00_Corpus/00_mihai_work/20190226_Production_selection_v0_3/master_labels')

#load pretrained model
tagger_model = tf.keras.models.load_model(logger.GetDropboxDrive() + '\_allan_data\_allan_tagger\_models\\allan_tagger_pretrained_model.h5')

welcome_message = "Hello!" 


#dummy example for testing
#print(predict_new_doc(d, "Am facut accident si ma doare dar clinica medicala a fost super si nu ma doare ma simt bine ce stare de bine am. ma simt super.", tagger_model))

text_banner = Paragraph(text=welcome_message, width=800, height=400)
 
#user_input list 
document_list = []

 # callback to predict on user_input 
def callback_print(args, old, new):
  if new:
    document_list.append(new)
    preds = predict_new_doc(d, new, tagger_model)
    text_banner.text = 'ALLAN found these labels' + preds
    text_input.value = ''

 # user interaction
text_input = TextInput(value="", title="Please introduce your own text for ALLAN to tag...")
text_input.on_change("value", callback_print)

 # layout
widget = widgetbox(text_input, text_banner)

curdoc().add_root(widget)

