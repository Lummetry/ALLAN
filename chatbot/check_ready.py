from libraries.logger import Logger
import os

if __name__ == '__main__':
  logger = Logger(lib_name='ALLANBOT',
                  config_file='chatbot/config_check_ready.py',
                  TF_KERAS=False)
  
  texts_folder = logger.GetDataSubFolder(logger.config_data['TEXTS_FOLDER'])
  labels_folder = logger.GetDataSubFolder(logger.config_data['LABELS_FOLDER'])
  all_labels_fn = logger.GetDataFile(logger.config_data['ALL_LABELS_FN'])
  user_labels_fn = os.path.join(logger.GetDataFolder(), logger.config_data['SAVE_USER_LABELS'])
  bot_labels_fn = os.path.join(logger.GetDataFolder(), logger.config_data['SAVE_BOT_LABELS'])

  
  logger.P("* Texts  folder: '{}'".format(texts_folder))
  logger.P("* Labels folder: '{}'".format(labels_folder))
  logger.P("* All labels fn: '{}'".format(all_labels_fn))
  
  conv_texts = set(os.listdir(texts_folder))
  conv_labels = set(os.listdir(labels_folder))
  
  set_all_labels = set()
  set_user_labels = set()
  set_bot_labels = set()
  with open(all_labels_fn, 'r') as f:
    lines = f.read().splitlines()
  set_all_labels.update(lines)
  
  nr_warnings = 0
  for x in conv_texts - conv_labels:
    logger.P(" [WARNING] File '{}' in texts, but not in labels".format(x))
    nr_warnings += 1
    
  for x in conv_labels - conv_texts:
    logger.P(" [WARNING] File '{}' in labels, but not in texts".format(x))
    nr_warnings += 1
  
  for x in conv_labels:
    path = os.path.join(labels_folder, x)
    with open(path, 'r') as f:
      lines = f.read().splitlines()
    for i in range(len(lines)):
      lbl = lines[i]
      if lbl not in set_all_labels:
          logger.P(" [WARNING] Label File '{}', row {} - label '{}' not found"
                   .format(x, i+1, lbl))
          nr_warnings += 1
      if i % 2 == 0:
        set_bot_labels.add(lbl)
      else:
        set_user_labels.add(lbl)
    #endfor
  #endfor
  
  str_log = 'Found {} warnings. '.format(nr_warnings)
  if nr_warnings > 0:
    str_log += 'Please review them and re-run.'
    logger.P(str_log)
  else:
    str_log += 'Everything OK.'
    logger.P(str_log)
    
    with open(user_labels_fn, 'w') as f:
      for lbl in set_user_labels:
        f.write('%s\n' % lbl)
    
    with open(bot_labels_fn, 'w') as f:
      for lbl in set_bot_labels:
        f.write('%s\n' % lbl)
    
    logger.P("* Saved user labels to: '{}'".format(user_labels_fn))
    logger.P("* Saved bot labels to: '{}'".format(bot_labels_fn))
