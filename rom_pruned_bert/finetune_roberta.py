"""
Copyright 2019 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


* NOTICE:  All information contained herein is, and remains
* the property of Knowledge Investment Group SRL.  
* The intellectual and technical concepts contained
* herein are proprietary to Knowledge Investment Group SRL
* and may be covered by Romanian and Foreign Patents,
* patents in process, and are protected by trade secret or copyright law.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Knowledge Investment Group SRL.


@copyright: Lummetry.AI
@author: Lummetry.AI
@project: 
@description:
"""

def test_model(model, tokenizer, strings):
  fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
  )
  
  for s in strings:
    log.P("<mask> candidates for '{}':".format(s))
    ret = fill_mask(s)
    for candidate in ret:
      log.P("* {}".format(candidate), noprefix=True)
  #endfor
  return
#enddef


from libraries import Logger
import argparse
import pickle
import os
from transformers import RobertaTokenizerFast, RobertaForMaskedLM,\
  LineByLineTextDataset, DataCollatorForLanguageModeling,\
  Trainer, TrainingArguments, pipeline

###############
COMPUTE_DATASET = False
fn_corpus = '20200120_corpus_merged'
out_folder = fn_corpus[:9] + 'hf_BPE_tokenizer'
#################



parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_folder", help="Base folder for storage",
                    type=str, default='dropbox')
parser.add_argument("-a", "--app_folder", help="App folder for storage",
                    type=str, default='_allan_data/_rowiki_dump')
parser.add_argument("-v", "--vocab_size", type=int, default=100_000)
parser.add_argument("-f", "--min_freq", type=int, default=2)
parser.add_argument("-m", "--model", type=str, default='RoBERTa')

args = parser.parse_args()
base_folder = args.base_folder
app_folder = args.app_folder
vocab_size = args.vocab_size
min_freq = args.min_freq
model = args.model

log = Logger(
  lib_name='LM',
  base_folder=base_folder,
  app_folder=app_folder,
  TF_KERAS=False
)

path_corpus = log.get_data_file(fn_corpus)
path_out_folder = os.path.join(log.get_data_folder(), out_folder)

log.P("Loading tokenizer and model from pretrained ...")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
model = RobertaForMaskedLM.from_pretrained("roberta-large")
log.P("", show_time=True)

if COMPUTE_DATASET:
  dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=path_corpus,
    block_size=128,
  )
  with open(os.path.join(path_out_folder, 'LineByLineTextDataset.pk'), 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
  
  with open(os.path.join(path_out_folder, 'LineByLineTextDataset.pk'), 'rb') as handle:
    dataset = pickle.load(handle)
#endif

data_collator = DataCollatorForLanguageModeling(
  tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

output_dir = os.path.join(log.get_models_folder(), 'roberta')
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True
)

list_s = ["Mi-am luat Tesla si ma dau cu ea prin <mask>.",
          "Azi mi-am luat <mask> si ma duc la cumparaturi."]


log.P("Before training ...")
test_model(model, tokenizer, list_s)

# trainer.train()

log.P("After training ...")
test_model(model, tokenizer, list_s)


