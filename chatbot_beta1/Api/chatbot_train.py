import json
import numpy as np
import nltk
import torch
import queue
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .chatbot_nltk_utils import get_wordnet_pos, tokenize, lem, bag_of_words    # Imports the functions from out utilities file
from .model import NeuralNet

def training( 
    batch_size = 32,
    num_epochs = 1000,
    hidden_size = 32,
    intents_file = "intents.json",
    learning_rate = 0.0001,
    message = None
    ):

    # Loads the intents file
    with open(intents_file, 'r', encoding='utf-8') as f:
        intents = json.load(f)

    all_words = []    # Will store unique lemmas
    tags = []
    xy = []    # Stores tuples of (tokens, tags)

    # Ignore these symbols
    ignore_words = ['?', '!', '.', ',', "'s", "'", '(', ')', '/', '"']

    # Loop through each sentence in our intents patterns
    for intent in intents['intents']:
        tag = intent['tag'] 
        tags.append(tag)    # Adds the tags to list

        for pattern in intent['patterns']:
            tokens = tokenize(pattern)    # Tokenizes the pattern
            xy.append((tokens, tag))    # Adds tokens and tags to xy as tuple

            pos_tags = nltk.pos_tag(tokens)    # Performs POS tagging on original tokens

            # Lemmatize using POS tags and add to all_words list, ignoring punctuation and symbols
            for word, pos_tag_val in pos_tags:
                if word not in ignore_words:
                    wordnet_tag = get_wordnet_pos(pos_tag_val)
                    lemma = lem(word, wordnet_tag)
                    all_words.append(lemma)    # Appends the valid lemma


    # Sorts the lemmatized words list and tags list and removes duplicates by turning them into sets
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Create the training data
    X_train = []
    y_train = []

    for (pattern_tokens, tag) in xy:
        bag = bag_of_words(pattern_tokens, all_words)    # X: Gets a bag of words for each pattern sentence (tokenized and lemmatized pattern sentences)
        X_train.append(bag)
        label = tags.index(tag)    # y: Gets class labels, doesn't need one-hot enconding since we use PyTorch's CrossEntropyLoss
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Hyperparameters
    input_size = len(all_words)
    output_size = len(tags)

    # Class for out dataset
    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, idx):
            return self.x_data[idx], self.y_data[idx]
        
        def __len__(self):
            return self.n_samples
        

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device, dtype=torch.int64)    # Ensures the correct data type required by nn.CrossEntropyLoss

            # Forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if message and isinstance(message, queue.Queue) and (epoch + 1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

    if message and isinstance(message, queue.Queue):
        print(f'Final loss, loss={loss.item():.4f}')
        message.put("[Training] Done")

    # Save the data
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE ="model.pth"
    torch.save(data, FILE)
    if message and isinstance(message, queue.Queue):
        print(f"Training Complete, File saved to {FILE}")