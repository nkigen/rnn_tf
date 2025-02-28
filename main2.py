import torch
import numpy as np

# Create a data loader for batching
from torch.utils.data import DataLoader, TensorDataset

# Load the text
with open('input.txt', 'r') as f:
    text = f.read()

# Create mappings from character to index and vice versa
chars = sorted(list(set(text)))  # Unique characters in the text
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

vocab_size = len(chars)

# Convert the text into integer indices
text_as_int = np.array([char_to_idx[c] for c in text])

SEQ_LENGTH = 100  # Length of input sequence for each training example
BATCH_SIZE = 64

# Split the dataset into sequences of length SEQ_LENGTH + 1
def create_dataset(text_as_int, seq_length):
    sequences = []
    targets = []
    for i in range(len(text_as_int) - seq_length):
        sequences.append(text_as_int[i:i+seq_length])
        targets.append(text_as_int[i+1:i+seq_length+1])
    return torch.tensor(sequences), torch.tensor(targets)

sequences, targets = create_dataset(text_as_int, SEQ_LENGTH)


dataset = TensorDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.rnn(x, h)  # RNN forward pass
        out = self.fc(out)  # Convert to logits for each character
        return out, h

# Model hyperparameters
embedding_dim = 256
hidden_dim = 1024
model = CharRNN(vocab_size, embedding_dim, hidden_dim, BATCH_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10
for epoch in range(EPOCHS):
    h = None  # Initialize hidden state to None
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()

        # Forward pass
        outputs, h = model(inputs)

        # Compute loss
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader)}')

def generate_text(model, start_char, length=500, temperature=1.0):
    model.eval()
    idx = char_to_idx[start_char]
    input_seq = torch.tensor([[idx]])

    # Initialize hidden state
    h = None
    generated_text = start_char

    for _ in range(length):
        # Forward pass to predict the next character
        output, h = model(input_seq, h)

        # Get the logits for the last character in the sequence
        logits = output[0, -1, :]

        # Apply temperature scaling for diversity
        logits = logits / temperature

        # Sample the next character
        probs = torch.nn.functional.softmax(logits, dim=0).detach()
        next_char_idx = torch.multinomial(probs, 1).item()

        # Append the character to the generated text
        generated_text += idx_to_char[next_char_idx]

        # Update the input for the next character
        input_seq = torch.tensor([[next_char_idx]])

    return generated_text

# Generate text starting with a seed character
print(generate_text(model, start_char='A', length=500))


