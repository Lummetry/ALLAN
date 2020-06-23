from libraries import Logger
import os
import argparse
import numpy as np
import shutil

from sklearn.model_selection import train_test_split

def get_fn_without_extension(fn):
  filename, file_extension = os.path.splitext(fn)
  return filename

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-b", "--base_folder", help="Base folder for storage",
                      type=str, default='dropbox')
  parser.add_argument("-a", "--app_folder", help="App folder for storage",
                      type=str, default='_allan_data/_chatbot')
  
  args = parser.parse_args()
  base_folder = args.base_folder
  app_folder = args.app_folder
  l = Logger(lib_name='ALLAN', base_folder=base_folder, app_folder=app_folder)
  l_t = Logger(lib_name='ALLAN', base_folder='.', app_folder='chatbot_v2')
  
  fld_workspace  = '20190506_Imobiliare_v06 WIP'
  fld_utterances = l.get_data_subfolder(os.path.join(fld_workspace,
                                                     'hashtags_utterances'))
  
  fld_tagger  = l_t.get_data_subfolder('_hashtags_utterances_tagger')
  fn_labels   = 'labels.txt'
  
  fn_all_utterances = os.listdir(fld_utterances)

  dct_labels = {'full' : {}, 'train' : {}, 'dev' : {}}
  for fn in fn_all_utterances:
    label = get_fn_without_extension(fn)
    
    with open(os.path.join(fld_utterances, fn), 'r') as f:
      lines = f.read().splitlines()  
    
    if len(lines) > 0:
      dct_labels['full'][label] = lines

  #endfor
  
  labels = np.array(list(dct_labels['full'].keys()))
  nr_obs_per_label = np.array([len(v) for k,v in dct_labels['full'].items()])
  _min, _q25, _q50, _q75, _q90, _max = np.quantile(nr_obs_per_label,
                                                   q=[0, 0.25, 0.5, 0.75, 0.9, 1])  

  l.P("Found {} labels:".format(len(labels)))
  l.P("  {}".format(list(labels)))
  l.P("Distribution of nr obs per label:")
  l.P("  min:{}  q25:{}  med:{}  q75:{}  q90:{}  max:{}"
      .format(_min, _q25, _q50, _q75, _q90, _max))
  
  soft_labels_idx = np.where(nr_obs_per_label < _q25)[0]
  
  soft_labels = labels[soft_labels_idx]
  soft_labels_nr_obs = nr_obs_per_label[soft_labels_idx]
  
  l.P("For the following labels, nr of examples is too small: {}"
      .format(list(zip(soft_labels, soft_labels_nr_obs))))

  for label in dct_labels['full']:
    nr_obs = len(dct_labels['full'][label])
    try:
      train, dev = train_test_split(dct_labels['full'][label],
                                    test_size=min(4, int(0.2 * nr_obs)))
      dct_labels['train'][label] = train
      dct_labels['dev'][label] = dev
    except:
      dct_labels['train'][label] = dct_labels['full'][label]
  
  
  
  dct_labels_to_descr = {}
  for lbl in dct_labels['full']:
    dct_labels_to_descr[lbl] = 'descriere_{}'.format(lbl)
  
  l_t.save_data_json(dct_labels_to_descr, fn_labels)
  
  
  
  
  if True:
    for ds in dct_labels.keys():
      fld_paste = os.path.join(fld_tagger, ds)
      fld_paste_texts = os.path.join(fld_paste, 'texts')
      fld_paste_labels = os.path.join(fld_paste, 'labels')
      
      shutil.rmtree(fld_paste_texts)
      shutil.rmtree(fld_paste_labels)
      
      os.mkdir(fld_paste_texts)
      os.mkdir(fld_paste_labels)
      
      cnt = 0
      for k,v in dct_labels[ds].items():
        for i in range(len(v)):
          cnt += 1
          with open(os.path.join(fld_paste_texts, "{}.txt".format(cnt)), 'w') as f:
            f.write(v[i] + '\n')
          with open(os.path.join(fld_paste_labels, "{}.label".format(cnt)), 'w') as f:
            f.write(k + '\n')
    
    