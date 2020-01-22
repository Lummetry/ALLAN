import os
import gensim
import argparse

from libraries.logger import Logger

def most_similar(word, n=15, show_dist=False):
  most_similar = model.wv.most_similar(word, topn=n)
  if not show_dist:
    most_similar = [x[0] for x in most_similar]
  log.P("Most similar {} words to {}:\n{}".format(n,word,most_similar))
  return

if __name__ == '__main__':
  
  log = Logger(lib_name='ALLANVOCAB', config_file='chatbot/word_universe/config.txt',
               TF_KERAS=False)
  
  models_folder = log.GetModelsFolder()
  models = os.listdir(models_folder)
  filtered_models = list(filter(lambda x: 'w2v' in x and '.' not in x,
                                models))

  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", help="Model that should be loaded from the _models folder",
                      type=str, default=filtered_models[-1])

  args = parser.parse_args()
  
  fn_model = os.path.join(models_folder, args.model)
  
  log.P("Loading model from {} ...".format(fn_model))
  model = gensim.models.Word2Vec.load()
  log.P(" Model loaded.", show_time=True)
