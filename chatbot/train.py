import json
import numpy as np
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from chatbot.nltk_utils import get_wordnet_pos, tokenize, lem, bag_of_words
from chatbot.model import NeuralNet

# File Paths
# Note: This script should be run from the main project directory,
# for example: python chatbot/train.py
INTENTS_FILE = "data/intents.json"
MODEL_FILE = "model.pth"
METRICS_FILE = "assets/training_metrics_1000epochs.png" # Saving with a consistent name

# Loads the intents file
with open(INTENTS_FILE, 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

ignore_words = ['?', '!', '.', ',', "'s", "'", '(', ')', '/', '"']

for intent in intents['intents']:
    tag = intent['tag'] 
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        xy.append((tokens, tag))
        # Note: nltk must be downloaded once per environment
        # nltk.download('averaged_perceptron_tagger') 
        pos_tags = nltk.pos_tag(tokens)
        for word, pos_tag_val in pos_tags:
            if word not in ignore_words:
                wordnet_tag = get_wordnet_pos(pos_tag_val)
                lemma = lem(word, wordnet_tag)
                all_words.append(lemma)

all_words = sorted(set(all_words))
tags = sorted(set(tags))

X = []
y = []
for (pattern_tokens, tag) in xy:
    bag = bag_of_words(pattern_tokens, all_words)
    X.append(bag)
    label = tags.index(tag)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Hyperparameters
batch_size = 32
hidden_size = 32
input_size = len(all_words)
output_size = len(tags)
learning_rate = 0.0001
num_epochs = 1000 # Increased to match your image name
validation_split = 0.2

class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.x_data = X_data
        self.y_data = y_data
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    def __len__(self):
        return self.n_samples

full_dataset = ChatDataset(X, y)
val_size = int(len(full_dataset) * validation_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s = [], [], [], [], [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for (words, labels) in train_loader:
        words, labels = words.to(device), labels.to(device, dtype=torch.int64)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * words.size(0)
    train_losses.append(train_loss / len(train_loader.dataset))

    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for (words, labels) in val_loader:
            words, labels = words.to(device), labels.to(device, dtype=torch.int64)
            outputs = model(words)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * words.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_losses.append(val_loss / len(val_loader.dataset))

    val_accuracies.append(accuracy_score(all_labels, all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}')

# Plotting and saving logic
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(METRICS_FILE) # Save to assets folder
plt.show()

data = {
    "model_state": model.state_dict(), "input_size": input_size,
    "output_size": output_size, "hidden_size": hidden_size,
    "all_words": all_words, "tags": tags
}

torch.save(data, MODEL_FILE)
print(f"Training Complete. Model saved to {MODEL_FILE}")
print(f"Metrics saved to {METRICS_FILE}")