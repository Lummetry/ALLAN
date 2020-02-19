from tagger.brain.allan_tagger_creator import ALLANTaggerCreator
from libraries.logger import Logger
from tagger.brain.data_loader import ALLANDataLoader

if __name__ == '__main__':
  cfg = 'tagger/brain/configs/20200212_config_ey_cv.txt'
  
  
  VALIDATION = True
    
  l = Logger(lib_name="ALNCV",config_file=cfg)
  l.SupressTFWarn()

  loader = ALLANDataLoader(log=l, multi_label=False, 
                           normalize_labels=False)
  loader.LoadData(exclude_list=['ï»¿'], remove_punctuation=True, save_dicts=False)
  
  valid_texts, valid_labels = None, None
  if VALIDATION:
    folder = l.config_data['TRAINING']['DEV_FOLDER']
    doc_folder, label_folder = None, None
    doc_ext = l.config_data['TRAINING']['DOCUMENT']
    label_ext = l.config_data['TRAINING']['LABEL']
    if l.config_data['TRAINING']['SUBFOLDERS']['ENABLED']:
      doc_folder = l.config_data['TRAINING']['SUBFOLDERS']['DOCS']
      label_folder = l.config_data['TRAINING']['SUBFOLDERS']['LABELS']
    
    valid_texts, valid_labels = l.LoadDocuments(folder=folder,
                                                doc_folder=doc_folder,
                                                label_folder=label_folder,
                                                doc_ext=doc_ext,
                                                label_ext=label_ext,
                                                return_labels_list=False,
                                                exclude_list=['ï»¿'])

  
  
  loader.encode(loader.raw_documents[0], convert_unknown_words=False)
  
#  epochs = 200
#  
#  model_def = l.config_data['MODEL']
#  model_name = model_def['NAME']
#  eng = ALLANTaggerCreator(log=l, 
#                           dict_word2index=loader.dic_word2index,
#                           dict_label2index=loader.dic_labels,
#                           dict_topic2tags=loader.dic_topic2tags)
#  
#  if VALIDATION:
#    new_valid_labels = eng.check_labels_set(valid_labels, exclude=True)
#  
#  eng.setup_model(dict_model_config=model_def, model_name=model_name) # default architecture
#  
#  hist = eng.train_on_texts(loader.raw_documents,
#                            loader.raw_labels,
#                            n_epochs=epochs,
#                            convert_unknown_words=True,
#                            save=True,
#                            X_texts_valid=valid_texts,
#                            y_labels_valid=new_valid_labels,
#                            skip_if_pretrained=False,
#                            DEBUG=False,
#                            test_top=1,
#                            compute_topic=False,
#                            sanity_check=False,
#                            batch_size=2,
#                            remove_punctuation=True)
#  l.show_timers()
  