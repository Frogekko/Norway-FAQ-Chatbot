# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 13:36:08 2025

@author: Fredrik
Nltk utils file
"""

import nltk

#nltk.download('punkt') # Remember to unhash these if it is youre fist time running this file
#nltk.download('punkt_tab') # Remember to unhash these if it is youre fist time running this file
from nltk.stem.snowball  import SnowballStemmer
import numpy as np

stemmer = SnowballStemmer("english") # Remember to select apropriet language

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag