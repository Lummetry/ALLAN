from libraries import Logger
import argparse
import os

from collections import Counter

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
  
  fld = '20190506_Imobiliare_v06 WIP'
  fld_texts = l.get_data_subfolder(os.path.join(fld, 'Texts_new'))
  fld_lbls  = l.get_data_subfolder(os.path.join(fld, 'Labels_new'))
  fn_hashtags_1 = l.get_data_file(os.path.join(fld, 'hashtags_spatii_birouri.txt'))
  fn_hashtags_2 = l.get_data_file(os.path.join(fld, 'hashtags_spatiu_locuit.txt'))
  fn_conv_texts = os.listdir(fld_texts)
  fn_conv_lbls  = os.listdir(fld_lbls)
  assert len(fn_conv_lbls) == len(fn_conv_texts)
  
  
  set_hashtags = set()
  with open(fn_hashtags_1, 'r') as f:
    set_hashtags.update(f.read().splitlines())
  
  with open(fn_hashtags_2, 'r') as f:
    set_hashtags.update(f.read().splitlines())
    
  dct_cnt_hashtags = {h:0 for h in set_hashtags}
  cnt_all_labels = Counter()

  l.P("* Texts  folder: '{}' ({} convs)".format(fld_texts, len(fn_conv_texts)))
  l.P("* Labels folder: '{}' ({} convs)".format(fld_lbls, len(fn_conv_lbls)))

  
  lst_bot_replicas = []
  for fn in fn_conv_texts:
    with open(os.path.join(fld_texts, fn), 'r') as f:
      lst_text_lines = f.read().splitlines()
    
    with open(os.path.join(fld_lbls, fn), 'r') as f:
      lst_lbls_lines = f.read().splitlines()
    
    if len(lst_text_lines) != len(lst_lbls_lines):
      l.P("WARNING! File '{}' has no matching texts - labels".format(fn))

    cnt_all_labels.update(Counter(lst_lbls_lines[1::2]))
    lst_bot_replicas += lst_text_lines[::2]

  dct_cnt_hashtags = {h: cnt_all_labels[h] for h in dct_cnt_hashtags}

    
    

  
  