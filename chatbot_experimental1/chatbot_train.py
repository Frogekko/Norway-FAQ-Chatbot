import json
import numpy as np
from chatbot_nltk_utils import tokenize, stem, bag_of_words    # Imports the functions from out utilities file

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag'] 
    tags.append(tag)    # Adds the tags to list
    pattern = intent['patterns']
    for pattern in intent['patterns']:
        word = tokenize(pattern)
        all_words.extend(word)    # Adds patterns to words list
        xy.append((word, tag))    # Adds to xy pair

# Ignore these symbols
ignore_words = ['?', '!', '.', ',', "'s", "'", '(', ')', '/', '"']

# Applies stemming to all_words list
all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))    # By turning this into a set we get rid of the duplicates in the list

tags = sorted(set(tags))    # Removes duplicates if there are any

# Test
print(all_words)
print(tags)

# Create the training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)    # X: Gets a bag of words for each pattern_sentence (tokenized and stemmed pattern sentences)
    X_train.append(bag)
    label = tags.index(tag)    # y: Gets class labels, doesn't need one-hot enconding since we use PyTorch's CrossEntropyLoss
    y_train.append(label)


