
from libraries.logger import Logger
import os
import random
import shutil

if __name__ == '__main__':
  
  log = Logger(lib_name='TAG', config_file='tagger/data/config.txt', TF_KERAS=False)
  
  data_folder = log.config_data['DATA_FOLDER']
  dev_size = log.config_data['DEV_SIZE']
  balance_classes = bool(log.config_data['BALANCE_CLASSES'])
  ext_label = log.config_data['EXT_LABEL']
  
  splitted_data_folders = {'train': log.config_data['TRAIN_DATA_FOLDER'],
                           'dev': log.config_data['DEV_DATA_FOLDER'],
                           'full': log.config_data['FULL_DATA_FOLDER']}
  
  full_data_folder = log.GetDataSubFolder(data_folder)
  
  log.P("Data Folder: '..{}'".format(full_data_folder[-50:]))

  dct_main_profiles = {}
  dct_main_profiles['train'] = {}
  dct_main_profiles['full'] = {}
  dct_main_profiles['dev'] = {}
    
  
  for k,v in splitted_data_folders.items():
    path = log.GetDataSubFolder(os.path.join(data_folder, k))
    if path:
      shutil.rmtree(path)
      log.P("Deleted {} path '..{}'.".format(k, path[-50:]))
  #endfor
  
  total_files = 0
  for root, sub_folders, files in os.walk(full_data_folder):
    sub_folders = list(filter(lambda x: not x.startswith('.'), sub_folders))
    files = list(filter(lambda x: not x.startswith('.'), files))
    files = list(filter(lambda x: x.endswith('.txt'), files))
    
    if root == full_data_folder:
      log.P("  Main profiles: {}".format(sub_folders))
      continue
    
    last_slash = root.replace('\\', '/').rfind('/')
    
    main_profile = root[last_slash+1:]    
    
    log.P("  Profile {:>15} has {:>3} files".format(main_profile[:15], len(files)))
    
    dct_main_profiles['full'][main_profile] = files
    total_files += len(files)
  
  log.P("Total files: {}".format(total_files))
  
  if balance_classes:
    nr_min_files = min([len(v) for v in dct_main_profiles['full'].values()])
    log.P("Balancing classes to {} examples per each class".format(nr_min_files))
    
    for k,v in dct_main_profiles['full'].items():
      dct_main_profiles['full'][k] = random.sample(v, nr_min_files)
  #endif
  
  for k, v in dct_main_profiles['full'].items():
    dev_files = random.sample(v, k=int(dev_size * len(v)))        
    train_files = list(set(v) - set(dev_files))
    
    dct_main_profiles['train'][k] = train_files
    dct_main_profiles['dev'][k] = dev_files
  #endfor
  
  total_files = 0
  total_train_files = 0
  total_dev_files = 0
  for k in dct_main_profiles['full'].keys():
    full_files = dct_main_profiles['full'][k]
    train_files = dct_main_profiles['train'][k]
    dev_files = dct_main_profiles['dev'][k]
    
    log.P("Profile {:>15} has {:>3} files ({:<3} train / {:<3} dev)".
          format(k[:15], len(full_files), len(train_files), len(dev_files)))
    
    total_files += len(full_files)
    total_train_files += len(train_files)
    total_dev_files += len(dev_files)
  #endfor
  log.P("Total files: {} / Train files: {} / Dev files: {}"
        .format(total_files, total_train_files, total_dev_files))
  

  log.P("Splitting data in folders ...")
  for dataset, v in splitted_data_folders.items():
    path = os.path.join(log.GetDataFolder(), data_folder, v)
    try:
      os.mkdir(path)
      os.mkdir(os.path.join(path, 'texts'))
      os.mkdir(os.path.join(path, 'labels'))
    except OSError:
      log.P("Creation of the {} data directory '..{}' FAILED".format(dataset, path))
    else:
      log.P("Successfully created the {} data directory '..{}'".format(dataset, path))
  
    dst_texts = os.path.join(full_data_folder, dataset, 'texts')
    dst_labels = os.path.join(full_data_folder, dataset, 'labels')
    for main_profile, files in dct_main_profiles[dataset].items():
      str_main_profile = 'profile_' + main_profile.replace(' ', '_')
      src = os.path.join(full_data_folder, main_profile)
      for f in files:
        ext_pos = f.rfind('.')
        f_no_ext = f[:ext_pos]
   
        shutil.copyfile(src=os.path.join(src, f),
                        dst=os.path.join(dst_texts, f))
        
        with open(os.path.join(dst_labels, f_no_ext + ext_label), 'w') as handle:
          handle.write(str_main_profile + '\n')
        
      log.P("  Copied {} files from '..{}' to '..{}' and created labels.".format(len(files), src[-30:], dst_texts[-30:]))
    
  
      
  

