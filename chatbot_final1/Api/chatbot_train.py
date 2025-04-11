import json
import numpy as np
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from chatbot_nltk_utils import get_wordnet_pos, tokenize, lem, bag_of_words
from model import NeuralNet
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Loads the intents file
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Ignore these symbols
ignore_words = ['?', '!', '.', ',', "'s", "'", '(', ')', '/', '"']

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag'] 
    tags.append(tag)

    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        xy.append((tokens, tag))

        pos_tags = nltk.pos_tag(tokens)

        for word, pos_tag_val in pos_tags:
            if word not in ignore_words:
                wordnet_tag = get_wordnet_pos(pos_tag_val)
                lemma = lem(word, wordnet_tag)
                all_words.append(lemma)

# Sorts the lemmatized words list and tags list and removes duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create the data
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
num_epochs = 1000
validation_split = 0.2    # 20% of data used for validation

# Class for our dataset
class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.x_data = X_data
        self.y_data = y_data

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
        
    def __len__(self):
        return self.n_samples

# Create full dataset
full_dataset = ChatDataset(X, y)

# Split into train and validation
val_size = int(len(full_dataset) * validation_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store metrics for plotting
train_losses = []
val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.int64)

        # Forward
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * words.size(0)
    
    # Calculate average training loss
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (words, labels) in val_loader:
            words = words.to(device)
            labels = labels.to(device, dtype=torch.int64)
            
            # Forward
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * words.size(0)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels for metric calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average validation loss
    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    
    # Calculate metrics
    val_accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    val_accuracies.append(val_accuracy)
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.4f}, Val F1: {f1:.4f}')

print(f'Final - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, '
      f'Val Acc: {val_accuracies[-1]:.4f}, Val F1: {val_f1s[-1]:.4f}')

# Plot training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs+1), val_accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')

# Plot precision, recall, and F1
plt.subplot(1, 3, 3)
plt.plot(range(1, num_epochs+1), val_precisions, label='Precision')
plt.plot(range(1, num_epochs+1), val_recalls, label='Recall')
plt.plot(range(1, num_epochs+1), val_f1s, label='F1')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

# Save the model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "model.pth"
torch.save(data, FILE)
print(f"Training Complete, File saved to {FILE}")
print(f"Metrics visualization saved to training_metrics.png")