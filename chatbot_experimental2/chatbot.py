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
import csv

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device.")

script_dir = os.path.dirname(os.path.abspath(__file__))

corpus_name = "Chat_Corpus"
corpus = os.path.join(script_dir, corpus_name)

def printLines(file, n=10):
    with open(file, "r", encoding="utf-8") as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "test_words.txt"))

def conversationalist(fileName):
    lines = {}
    conversations = {}
    with open(fileName, "r", encoding="utf-8") as f:
        for line in f:
            lineJson = json.loads(line)
            #Extraction fields
            lineOBJ = {}
            lineOBJ["lineID"] = lineJson["id"]
            lineOBJ["characterID"] = lineJson["speaker"]
            lineOBJ["text"] = lineJson["text"]
            lines[lineOBJ["lineID"]] = lineOBJ

            if lineJson["convo_id"] not in conversations:
                convOBJ = {}
                convOBJ["convoID"] = lineJson["convo_id"]
                convOBJ[] = lineJson[""][""]
                convOBJ[""] = [lineOBJ]
            else:
                convOBJ = conversations[lineJson["convo_id"]]
                convOBJ[""].insert(0, lineOBJ)
            conversations[convOBJ["convoID"]] = convOBJ
    
    return lines, conversations

def extraction(conversations):
    qa_pairs = []
    for conversation in conversations.values():
        # iterate all lines in convo
        for i in range(len(conversation[""]) - 1):
            inputLine = conversation[""][i][""].strip()
            targetLine = conversation[""][i+1][""].strip()
            #filter wrong samples
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


datafile = os.path.join(corpus, "test_words.txt")

delimiter = "\t"

delimiter = str(codecs.decode(delimiter, "unicode_escape"))

lines = {}
conversations = {}

print("\nWriting newly formatted filee...")
with open(datafile, "w", encoding = "utf-8") as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator="\n")
    for pair in extraction(conversations):
        writer.writerow(pair)

print("\nSample lines from file:")
printLines(datafile)