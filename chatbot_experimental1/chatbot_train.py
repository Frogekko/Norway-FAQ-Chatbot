import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from chatbot_nltk_utils import tokenize, stem, bag_of_words    # Imports the functions from out utilities file
from model import NeuralNet

# Loads the intents file
with open('intents.json', 'r', encoding='utf-8') as f:
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

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
batch_size = 32
hidden_size = 32
input_size = len(all_words)
output_size = len(tags)
learning_rate = 0.0001
num_epochs = 1000

# Test
print(input_size)
print(len(all_words))
print(output_size)
print(len(tags))

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

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'Final loss, loss={loss.item():.4f}')

# Save the data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE ="data.pth"
torch.save(data, FILE)

print(f"Training Complete, File saved to {FILE}")