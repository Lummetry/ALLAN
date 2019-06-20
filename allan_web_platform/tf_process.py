from libraries.logger import Logger
from allan_web_platform.Runner import ChatBot
from time import sleep
import os
import pickle
from allan_web_platform.doc_utils import DocUtils

def get_message(folder):
  while True:
    sleep(1)
    listdir = os.listdir(folder)
    for file in listdir:
      if '_message.pickle' in file:
        return file

if __name__ == '__main__':
  config_file = 'config_runner.txt'
  pipe_folder = 'pipe'
  LEN_PREFIX = 15

  logger = Logger(lib_name="CHATBOT",
                  config_file=config_file,
                  TF_KERAS=True,
                  HTML=True)
  
  data_utils = DocUtils(logger.config_data['INDEX2WORD'],
                        max_nr_words=logger.config_data['MAX_WORDS'],
                        max_nr_chars=logger.config_data['MAX_CHARACTERS'])
  
  c = ChatBot(logger)
  c.LoadModels()
  
  logger.P("You can start the web application.")

  while True:
    msg_file = get_message(pipe_folder)
    with open(os.path.join(pipe_folder, msg_file), 'rb') as handle:
      message_history = pickle.load(handle)
    os.remove(os.path.join(pipe_folder, msg_file))
    prefix = msg_file[:LEN_PREFIX]

    try:
      reply, label = c._step_by_step_prediction(data_utils, message_history, method='argmax')
    except:
      reply, label = "A fost intampinata o eroare! Te rugam reincarca pagina.", ""

    with open(os.path.join(pipe_folder, prefix + '_response.pickle'), 'wb') as handle:
      pickle.dump((reply, label), handle, protocol=pickle.HIGHEST_PROTOCOL)
  #endwith