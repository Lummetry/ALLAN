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

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from libraries import Logger
import argparse
import os
import json
import pickle

from transformers import RobertaTokenizerFast, DistilBertTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling


def txt_to_tokens(tok, sent):
  return tok.convert_ids_to_tokens(tok.encode(sent))
  



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

log = Logger(lib_name='LM_BERT', base_folder=base_folder, app_folder=app_folder, TF_KERAS=False)

#################
TRAIN_TOKENIZER = False
COMPUTE_DATASET = False
fn_corpus = '20200120_corpus_merged'
out_folder = fn_corpus[:9] + 'hf_BPE_tokenizer'
#################

path_corpus = log.get_data_file(fn_corpus)
path_out_folder = os.path.join(log.get_data_folder(), out_folder)

if TRAIN_TOKENIZER:
  log.P("Training ByteLevelBPETokenizer on corpus @ '{}' with vocab_sz={} and min_freq={} ..."
        .format(path_corpus, vocab_size, min_freq))
  # Initialize a tokenizer
  tokenizer = ByteLevelBPETokenizer()
  
  # Customize training
  tokenizer.train(files=fn_corpus, vocab_size=vocab_size, min_frequency=min_freq, special_tokens=[
      "<s>",
      "<pad>",
      "</s>",
      "<unk>",
      "<mask>",
  ])
  log.P("  Finished training", show_time=True)
  
  if not os.path.exists(path_out_folder):
    os.makedirs(path_out_folder)
  
  tokenizer.save(path_out_folder)
  log.P("Saved the trained tokenizer @ '{}'".format(path_out_folder))
  
  del tokenizer
#endif

log.P("Loading RobertaTokenizerFast from '{}' ...".format(path_out_folder))

tokenizer1 = RobertaTokenizerFast.from_pretrained(path_out_folder, max_len=512)

ids1 = tokenizer1.encode('mouse logitech impreuna cu cana star wars')
tokens1 = tokenizer1.convert_ids_to_tokens(ids1)
print(tokens1)

if False:
  ## TODO run also with DistilBertTokenizerFast / BertTokenizer
  tokenizer2 = DistilBertTokenizerFast.from_pretrained(path_out_folder, max_len=512)
  ids2 = tokenizer2.encode('mouse logitech impreuna cu cana star wars')
  tokens2 = tokenizer2.convert_ids_to_tokens(ids2)
  print(tokens2)
#endif

tokenizer = tokenizer1


if True:
  config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
  )
  
  #As we are training from scratch, we only initialize from a config, 
  #not from an existing pretrained model or checkpoint.
  model = RobertaForMaskedLM(config=config)

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
  
  from transformers import Trainer, TrainingArguments
  output_dir = os.path.join(log.get_models_folder(), 'roberta')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  training_args = TrainingArguments(
      output_dir=output_dir,
      overwrite_output_dir=True,
      num_train_epochs=1,
      per_gpu_train_batch_size=8,
      save_steps=10_000,
      save_total_limit=2,
  )
  
  trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=dataset,
      prediction_loss_only=True,
  )
  
  log.P("Starting training...", color='g')
  trainer.train()


if False:
  
  models = {
    'DistilBERT' : 
      {
      "architectures" : [
        'DistilBertForMaskedLM'
      ],
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
    	"hidden_dropout_prob": 0.1,
    	"hidden_dim": 4*768,
      "dim": 768,
    	"initializer_range": 0.02,
    	"layer_norm_eps": 1e-05,
    	"max_position_embeddings": 514,
    	"model_type": 'distilbert',
    	"num_attention_heads": 12,
    	"num_hidden_layers": 6,
    	"type_vocab_size": 1,
    	"vocab_size": vocab_size
      },
      
    'MobileBERT' : ('MobileBertForMaskedLM', 'mobilebert'),
    'RoBERTa'    : 
      {"architectures": [
      		'RobertaForMaskedLM'
      	],
      	"attention_probs_dropout_prob": 0.1,
      	"hidden_act": "gelu",
      	"hidden_dropout_prob": 0.1,
      	"hidden_size": 768,
      	"initializer_range": 0.02,
      	"intermediate_size": 3072,
      	"layer_norm_eps": 1e-05,
      	"max_position_embeddings": 514,
      	"model_type": 'roberta',
      	"num_attention_heads": 12,
      	"num_hidden_layers": 6,
      	"type_vocab_size": 1,
      	"vocab_size": vocab_size
      }
  }
  
  if True:
    log.P("Training model {} ...".format(model))
    output_dir = os.path.join(log.get_models_folder(), model)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
    config = models[model]
    	
    with open(os.path.join(path_out_folder, "config.json"), 'w') as fp:
      json.dump(config, fp)
        
    # tokenizer_config = {
    # 	"max_len": 512
    # }
    # with open(os.path.join(path_out_folder, "tokenizer_config.json"), 'w') as fp:
    #     json.dump(tokenizer_config, fp)
    
    import subprocess
    
    cmd = """
    python run_language_modeling.py
      --train_data_file {}
      --output_dir {}
      --model_type roberta
      --mlm
      --config_name {}
      --tokenizer_name {}
      --do_train
      --line_by_line
      --learning_rate 1e-4
      --num_train_epochs 1
      --save_total_limit 2
      --save_steps 2000
      --per_gpu_train_batch_size 16
      --seed 42
    """.replace("\n", " ").format(path_corpus, output_dir, path_out_folder, path_out_folder, )
    print(cmd)
    print('\n')
    
    # subprocess.call(cmd)

  
  

