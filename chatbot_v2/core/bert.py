from libraries.nlp import RomBERT
from libraries import Logger


if __name__ == '__main__':
  
  log = Logger(lib_name='BERT', base_folder='.', app_folder='chatbot_v2')
  eng = RomBERT(log=log, max_sent=50)
  # n = eng.text2embeds(['Mara are pere', 'Ana are mere'] * 5)
