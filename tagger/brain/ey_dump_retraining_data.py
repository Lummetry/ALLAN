import pandas as pd
import os

queries = []
intents = []

#### INPUTS ####
f1 = 'D:/Lummetry.AI Dropbox/DATA/_allan_data/_allan_tagger/_data/EY_FAQ/v3_equals_v2_and_single_tag/20190927_Validation_Texts'
f2 = 'D:/Lummetry.AI Dropbox/DATA/_allan_data/_allan_tagger/_data/EY_FAQ/v3_equals_v2_and_single_tag/20190927_Validation_Labels'
last_text = -1
fn_analysis = 'D:/Lummetry.AI Dropbox/DATA/_allan_data/_allan_tagger/_data/EY_FAQ_RAW_DATA/Answer Analisys_25.09.xlsx'
################

df = pd.read_excel(fn_analysis)

queries += df.Query.tolist()
intents += df['Correct intent'].tolist()

queries = [x for x in queries if str(x) != 'nan']
intents = [x for x in intents if str(x) != 'nan']

assert len(queries) == len(intents)

for i in range(len(queries)):
    q = queries[i]
    l = intents[i]
    nr = str(last_text + i + 1).zfill(4)
    fn = 'Text_{}.txt'.format(nr)
    with open(os.path.join(f1,fn), 'w', encoding='utf-8') as handle:
        handle.write(q)
    with open(os.path.join(f2,fn), 'w', encoding='utf-8') as handle:
        handle.write(l)