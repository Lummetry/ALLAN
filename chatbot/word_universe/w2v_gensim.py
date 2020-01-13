from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import time

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.start_time = time.time()

    def on_epoch_begin(self, model):
        self.start_time = time.time()
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        elapsed_time = time.time() - self.start_time
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print("Epoch #{} end. Loss: {} Elapsed time: {}".format(self.epoch,
                            model.get_latest_training_loss(),
                            time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        self.epoch += 1
#        model.save("gensim_w2v/gensim_w2v.model")


if __name__ == "__main__":
    sentences = []
    with open('/Users/laurentiupiciu/Lummetry.AI Dropbox/DATA/_allan_data/_rowiki_dump/_data/20200113_rowiki', "rt", encoding='utf8') as f:
        for line in f.readlines():
            sentences.append(line.split())

    epoch_logger = EpochLogger()

    model = Word2Vec(sentences=sentences,
                    iter=5,
                    sg=1,
                    alpha=0.005,
                    min_count=15,
                    window=2,
                    size=128,
                    negative=25,
                    workers=12,
                    compute_loss=True,
                    callbacks=[epoch_logger])
  
    """
    model.save(path + name)
    import pickle
    with open(path + name + '.index2word.pickle', 'wb') as handle:
      pickle.dump(model.wv.index2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model.train(sentences=sentences, total_examples=model.corpus_count,
                epochs=20,
                start_alpha=0.0025, compute_loss=True, callbacks=[epoch_logger])
    """
  
    
    
    
    