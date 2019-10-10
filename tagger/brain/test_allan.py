import os
import requests
from tqdm import tqdm
import pandas as pd

def populate_pairs(pairs, f_texts, f_labels):
  files = list(zip(os.listdir(f_texts), os.listdir(f_labels)))

  for (fn_txt, fn_lbl) in files:
    with open(os.path.join(f_texts, fn_txt), 'r', encoding='utf-8') as f:
      txt = f.read().splitlines()
    if len(txt) > 1:
      print(fn_txt)
      continue
    txt = txt[0]
    
    with open(os.path.join(f_labels, fn_lbl), 'r', encoding='utf-8') as f:
      lbl = f.read().splitlines()
    assert len(lbl) == 1
    lbl = lbl[0]
    assert 'topic_' in lbl
    
    pairs.append((txt,lbl))
  
  return pairs


def populate_pairs_v2(fn, sheet_idx=0):
  df = pd.read_excel(fn, sheet_name=sheet_idx)
  queries = df.Query.tolist()
  intents = df['Correct Intent'].tolist()
  
  queries = [x for x in queries if str(x) != 'nan']
  intents = [x for x in intents if str(x) != 'nan']
  
  assert len(queries) == len(intents)
  return list(zip(queries, intents))


pairs = []
if False:
  pairs = populate_pairs(pairs=pairs,
                         f_texts='D:/Lummetry.AI Dropbox/DATA/_allan_data/_allan_tagger/_data/EY_FAQ/v4_softmax/Texts',
                         f_labels='D:/Lummetry.AI Dropbox/DATA/_allan_data/_allan_tagger/_data/EY_FAQ/v4_softmax/Labels')

if True:
  pairs = populate_pairs_v2('D:/Lummetry.AI Dropbox/DATA/_allan_data/_allan_tagger/_data/EY_FAQ_RAW_DATA/AnswerAnalysis_2.10.xlsx',
                            sheet_idx=1)


results_train = {
    'input': [],
    'pre_topic_document': [],
    'pred_topic': [],
    'true_topic': [],
    'score': []
}

results_val = {
    'input': [],
    'pred_topic': [],
    'true_topic': [],
    'score': []
    }

for p in tqdm(pairs):
  response = requests.post('http://127.0.0.1:5001/analyze',
                           data={'current_query': p[0]})
  
  results_train['input'].append(p[0])
  results_train['pre_topic_document'].append(response.json()['result']['pre_topic_document'])
  results_train['true_topic'].append(p[1])
  results_train['pred_topic'].append(response.json()['result']['topic_document'])
  results_train['score'].append(response.json()['result']['topic_score'])

if False:
  pairs = []
  pairs = populate_pairs(pairs=pairs,
                         f_texts='D:/Lummetry.AI Dropbox/DATA/_allan_data/_allan_tagger/_data/EY_FAQ/v4_softmax/Validation_Texts',
                         f_labels='D:/Lummetry.AI Dropbox/DATA/_allan_data/_allan_tagger/_data/EY_FAQ/v4_softmax/Validation_Labels')
  
  
  for p in tqdm(pairs):
    response = requests.post('http://127.0.0.1:5000/analyze',
                             data={'current_query': p[0]})
    
    results_val['input'].append(p[0])
    results_val['true_topic'].append(p[1])
    results_val['pred_topic'].append(response.json()['result']['topic_document'])
    results_val['score'].append(response.json()['result']['topic_score'])
  
  df_val = pd.DataFrame(results_val)


df_train = pd.DataFrame(results_train)

def process_df(df):
  df_true = df[df['true_topic'] == df['pre_topic_document']]
  df_false = df[df['true_topic'] != df['pre_topic_document']]
  
  recall = df_true.shape[0] / df.shape[0]

  print("Recall: {}".format(recall))
  print("TP score distribution:\n{}".format(df_true.score.describe().to_string()))
  print("FN score distribution:\n{}".format(df_false.score.describe().to_string()))    

