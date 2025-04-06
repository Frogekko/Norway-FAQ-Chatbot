import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import json
import math
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device.")

script_dir = os.path.dirname(os.path.abspath(__file__))

corpus_name = "Chat_Corpus"
corpus = os.path.join(script_dir, corpus_name)

def printLines(file, n=10):
    with open(file, "r", encoding="iso_8859-1") as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "test_words.txt"))

def conversationalist(fileName):
    lines = {}
    conversations = {}
    with open(fileName, "r", encoding="iso-8859-1") as f:
        for line in f:
            lineJson = json.loads(line)