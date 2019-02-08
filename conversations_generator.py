#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:02:23 2018
@author: denisilie94
"""

import os
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from logger import Logger
from glob import glob
from nltk.tokenize import word_tokenize


class ConversationsUtils(object):

    def __init__(self, logger, window_size=50):
        self.logger = logger
        self.dictionary 	  = {}
        self.invDictionary    = {}
        
        self.deletedFiles = 0
        self.size_of_dict = 0
        self.window_size  = window_size

        self.config_data  = logger.config_data
        self._base_folder = logger._base_folder
        self.max_words    = logger.config_data["MAX_WORDS"]

        dictionary_path    = logger.config_data["WORDS_DICTIONARY"]
        with open(self.logger.GetDataFile(dictionary_path), 'rb') as handle:
          self.dictionary    = pickle.load(handle)

        invDictionary_path = logger.config_data["INV_WORDS_DICTIONARY"]
        with open(self.logger.GetDataFile(invDictionary_path), 'rb') as handle:
          self.invDictionary = pickle.load(handle)

        self.size_of_dict = len(self.dictionary)
        
        self.end_char_id = self.dictionary['<END>']
        self.start_char_id = self.dictionary['<START>']
    
    def _parse_config(self):
        self.path_dataset = logger.config_data["DATASET_FOLDER"]

    def update_dictionary(self, top_wiki=-1):
        '''
        Generate vocubalary for entire wikidump
        '''
        if '<UNK>' not in self.dictionary:
            # Add UNK to dictionary
            self.dictionary['<UNK>'] 		     	 = self.size_of_dict
            self.invDictionary[self.size_of_dict]    = '<UNK>'
            self.size_of_dict 				        += 1
        else:
            print('<UNK> already included in dictionary!!')

        self.padding = self.size_of_dict
        self.start   = self.size_of_dict + 1
        self.end     = self.size_of_dict + 2

        self.dictionary['<PAD>']   = self.padding
        self.dictionary['<START>'] = self.start
        self.dictionary['<END>']   = self.end

        self.invDictionary[self.padding] = '<PAD>'
        self.invDictionary[self.start]   = '<START>'
        self.invDictionary[self.end]     = '<END>'
        
        self.size_of_dict = len(self.dictionary)

        print("Dictionary generated")


    def generator_sentences(self):

        csv = open('diff_godot_words.csv','w')
        contor_unk = 0
        contor_kn  = 0
        all_words = []
        all_paths = [y for x in os.walk(self.path_dataset) for y in glob(os.path.join(x[0], '*.txt'))]       
        
        for file in all_paths[::-1]:
            f = open(file, 'r', encoding = "ISO-8859-1")
            text = f.read()
            
            sentences = text.splitlines()
            sentences = [s for s in sentences if s]
            
            words = word_tokenize(sentences[0])
            all_words.extend(words)
            gen_result = []
            for w in words:
                if w in self.dictionary:
                    contor_kn += 1
                    gen_result = gen_result + [self.dictionary[w]]
                else:
                    contor_unk += 1
                    gen_result = gen_result + [self.dictionary['<UNK>']] # UNK id = len(self.dictionary)
                    csv.write(w + '\n')

            gen_result = [gen_result]
            max_dim = len(gen_result[0])

            for s in sentences[1:]:
                words = word_tokenize(s)
                all_words.extend(words)
                vector_gen = []
                for w in words:
                    if w in self.dictionary:
                        contor_kn += 1
                        vector_gen = vector_gen + [self.dictionary[w]]
                    else:
                        contor_unk += 1
                        vector_gen = vector_gen + [self.dictionary['<UNK>']] # UNK id = len(self.dictionary)
                        csv.write(w + '\n')

                #get max size in input data
                max_dim = max(len(gr) for gr in gen_result)

                # adding new sentence in generator result
                vector_gen = [self.start] + vector_gen + [self.end]

                # padding with -1 | len(self.dictionary)+1
                if self.max_words != None:
                    max_dim = max(max_dim, self.max_words)

                for i,gr in enumerate(gen_result):
                    gr_padd =  [self.padding] * (max_dim - len(gr))
                    gen_result[i] = gr + gr_padd

                    if self.max_words != None:
                        del gen_result[i][self.max_words:]

                yield (np.array(gen_result), np.array(vector_gen))
                # remove start & end
                vector_gen = vector_gen[1:-1]
                gen_result = gen_result + [vector_gen]

                if len(gen_result) > self.window_size:
                    del gen_result[:len(gen_result) - self.window_size]

            f.close()
            print("Contor UNK: " + str(contor_unk))
            print("Contor KN: " + str(contor_kn))
        csv.close()

        all_words = list(set(all_words))
        self.godot_dictionary = dict()
        for word in all_words:
            if word in self.dictionary:
                self.godot_dictionary[word] = self.dictionary[word]
            else:
                self.godot_dictionary[word] = self.dictionary['<UNK>']
        pickle.dump(self.godot_dictionary, open('godot_dictionary.pickle','wb'))


    def input_word_text_to_tokens(self, lines):        
        tokens = []
        if type(lines) is not list:
          lines  = lines.splitlines()

        for line in lines:
            idText = []
            words = word_tokenize(line)
            
            if '<' in words: words.remove('<')
            if '>' in words: words.remove('>')
            if 'UNK' in words:
              idx = words.index('UNK')
              words[idx] = '<UNK>'
                        
            for word in words:
                idText.append(self.dictionary[word])

            idText_pad = [self.dictionary['<PAD>']] * (self.max_words - len(idText))
            idText = idText + idText_pad
            tokens.append(idText) 

        return np.array(tokens)


    def input_word_tokens_to_text(self, list_tokens):
        list_tokens = np.array(list_tokens)
        text = []
        rows, cols = list_tokens.shape
        
        for r in range(rows):
            row_text = ''
            for c in range(cols):
                if self.invDictionary[list_tokens[r][c]] != '<PAD>':
                    row_text = row_text + ' ' + self.invDictionary[list_tokens[r][c]]
            text = text + [row_text]
        return " ".join(text)


    def translate_generator_sentences(self, batch):
        in_batch  = batch[0]
        out_batch = batch[1]

        # print input
        for sentence in in_batch:
            new_sentence = [self.invDictionary[word] for word in sentence]
            print(new_sentence)

        # print output
        new_sentence = [self.invDictionary[word] for word in out_batch]
        print(new_sentence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file in JSON format", required=True)
    args = vars(parser.parse_args())
    config_file = args['config']
    
    logger = Logger(lib_name='ConversastionsUtils', config_file=config_file, TF_KERAS=False)

    convUtils  = ConversationsUtils(logger=logger, window_size=3)
    gen = convUtils.generator_sentences()


    all_batches = []
    for result in tqdm(gen):
        #godot.translate_generator_sentences(result)
        all_batches.append(result)
        #break
        pass

    # save all batches in pickle
    pickle.dump(all_batches, open('all_batches_godot.pickle', 'wb'))
