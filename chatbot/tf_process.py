from libraries.logger import Logger
from time import sleep
import os
import pickle
from models_creator.doc_utils import DocUtils
from models_creator.hierarchical_wrapper_multioutput import HierarchicalNet

def get_message(folder):
  while True:
    sleep(1)
    listdir = os.listdir(folder)
    for file in listdir:
      if '_message.pickle' in file:
        return file

if __name__ == '__main__':
  config_file = 'chatbot/config_runner.txt'
  pipe_folder = 'chatbot/server/pipe'
  
  if not os.path.exists(pipe_folder):
    os.makedirs(pipe_folder)
  
  LEN_PREFIX = 15

  logger = Logger(lib_name="ALLANBOT-RUNNER",
                  config_file=config_file,
                  TF_KERAS=True,
                  HTML=True)
  
  params_file = logger.config_data['PARAMS_FILE']
  model_file = logger.config_data['MODEL_FILE']
  assert params_file != "" and model_file != ""
  params = logger.LoadPickleFromOutput(params_file)

  data_utils = DocUtils(logger,
                        params['DATA_W2V_INDEX2WORD'],
                        max_nr_words=params['MAX_WORDS'],
                        max_nr_chars=params['MAX_CHARACTERS'],
                        dict_user_id2label=params['DICT_USER_ID2LABEL'],
                        dict_bot_id2label=params['DICT_BOT_ID2LABEL'])
  
  hnet = HierarchicalNet(logger, data_utils)
  hnet.LoadModelWeightsAndConfig(model_file)
  
  logger.P("You can start the web application.")

  while True:
    msg_file = get_message(pipe_folder)
    with open(os.path.join(pipe_folder, msg_file), 'rb') as handle:
      message_history = pickle.load(handle)
    os.remove(os.path.join(pipe_folder, msg_file))
    prefix = msg_file[:LEN_PREFIX]

    try:
      reply, label = hnet._step_by_step_prediction(message_history, method='argmax')
    except:
      reply, label = "A fost intampinata o eroare! Te rugam reincarca pagina.", ""

    with open(os.path.join(pipe_folder, prefix + '_response.pickle'), 'wb') as handle:
      pickle.dump((reply, label), handle, protocol=pickle.HIGHEST_PROTOCOL)
  #endwith