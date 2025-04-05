import json
from chatbot_nltk_utils import tokenize

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    pattern = intents['patterns']
    w = tokenize(pattern)
    all_words.extend(w)
    xy.append((w, tag))
