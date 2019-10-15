from libraries.logger import Logger
import os

def check_labels(log, dct, is_bot):
  name = 'BOT' if is_bot else 'USER'
  
  str_log = "[INFO] These are the labels used by the {}: {}\n\n".format(name, list(dct.keys())) 
  str_log += "Type names of labels in order to see the name of the files in which you can find these labels. "
  str_log += "If you want to stop, type 'q!'"
  log.P(str_log)
  while True:
    inp = input()
    if 'q!' in inp:
      break
    
    if inp not in dct:
      print("Not found in user labels!")
    else:
      print(dct[inp])
      
  return

if __name__ == '__main__':
  logger = Logger(lib_name='ALLANBOT',
                  config_file='chatbot/config_check_ready.py',
                  TF_KERAS=False)
  
  texts_folder = logger.GetDataSubFolder(logger.config_data['TEXTS_FOLDER'])
  labels_folder = logger.GetDataSubFolder(logger.config_data['LABELS_FOLDER'])
  all_labels_fn = logger.GetDataFile(logger.config_data['ALL_LABELS_FN'])
  user_labels_fn = os.path.join(logger.GetDataFolder(), logger.config_data['SAVE_USER_LABELS'])
  bot_labels_fn = os.path.join(logger.GetDataFolder(), logger.config_data['SAVE_BOT_LABELS'])
  check_user_bot_labels = bool(logger.config_data['CHECK_USER_BOT_LABELS'])
  
  conv_texts = set(os.listdir(texts_folder))
  conv_labels = set(os.listdir(labels_folder))
  
  logger.P("* Texts  folder: '{}' ({} convs)".format(texts_folder, len(conv_texts)))
  logger.P("* Labels folder: '{}' ({} convs)".format(labels_folder, len(conv_labels)))
  logger.P("* All labels fn: '{}'".format(all_labels_fn))
  
  set_all_labels = set()
  dct_user_labels = dict()
  dct_bot_labels = dict()
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
        if lbl not in dct_bot_labels:
          dct_bot_labels[lbl] = []
        dct_bot_labels[lbl].append(x)
      else:
        if lbl not in dct_user_labels:
          dct_user_labels[lbl] = []
        dct_user_labels[lbl].append(x)
    #endfor
  #endfor
  
  set_user_labels = set(list(dct_user_labels.keys()))
  set_bot_labels = set(list(dct_bot_labels.keys()))
  user_bot_intersection = set_user_labels & set_bot_labels
  
  if len(user_bot_intersection) > 0:
    logger.P("", noprefix=True)
    logger.P("[INFO] The following labels are used both for BOT and USER: {}"
             .format(list(user_bot_intersection)))
  
  not_used_labels = set_all_labels - set(list(set_user_labels) + list(set_bot_labels))
  str_log = '[INFO] The following labels are not used yet: '
  for x in not_used_labels:
    str_log += "'{}', ".format(x)
  str_log = str_log[:-2]
  logger.P("", noprefix=True)
  logger.P(str_log)
  logger.P("", noprefix=True)
  
  if check_user_bot_labels:
    check_labels(logger, dct_user_labels, is_bot=False)
    check_labels(logger, dct_bot_labels, is_bot=True)
  
  for k,v in dct_user_labels.items():
    dct_user_labels[k] = len(set(v))
  for k,v in dct_bot_labels.items():
    dct_bot_labels[k] = len(set(v))
  
  logger.P("", noprefix=True)
  logger.P("[INFO] Number of conversations in which the following USER labels are used: {}"
           .format(list(dct_user_labels.items())))
  logger.P("", noprefix=True)
  logger.P("[INFO] Number of conversations in which the following BOT labels are used: {}"
           .format(list(dct_bot_labels.items())))
  logger.P("", noprefix=True)
  
  
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
  

