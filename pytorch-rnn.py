import random
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import Counter

characters = " ".join("abcdefghijklmnopqrstuvwxyz").split()
characters.append(" ")
alphabet = " ".join("abcdefghijklmnopqrstuvwxyz").split()
values = " ".join("abcdefghijklmnopqrstuvwxyz").split()
random.shuffle(values)
embedding_dim = 10
hidden_dim = 10
vocab_size = len(characters)
num_epochs = 10
accuracies, max_accuracy = [], 0
num_examples = 30

key_dict = dict(zip(alphabet, values))

def encode(message, key_dict):
    cipher = ''
    for letter in message:
        if letter != ' ':
            cipher += key_dict[letter]
        else:
            cipher += ' '

    return cipher

def data_gen():
    dataset = []
    with open ("PlainText.txt", "r") as myfile:
        file=myfile.readlines()
    del file[-355]
    for line in file:
        # changes cipher every time
        # random.shuffle(alphabet)
        # key_dict = dict(zip(alphabet, values))
        print(key_dict)
        line = re.sub(r"[^a-zA-Z ]","",line)
        line = re.sub(r"^ +","",line)
        line = line.lower()
        if len(line) != 0:
            ex_out = line
            ex_in = encode(line, key_dict)
            print(ex_out)
            print(ex_in)
            ex_in = [characters.index(x) for x in ex_in]
            # may be: [25, 13, 26, 3, 12, 5, 2, 26, 26, ...
            ex_out = [characters.index(x) for x in ex_out]
            # may be: [12, 0, 13, 17, 26, 19, 16, 13, ...
            dataset.append([torch.tensor(ex_in), torch.tensor(ex_out)])
        else:
            pass

    return dataset

dataset = data_gen()

def zero_hidden():
    return (torch.zeros(1, 1, hidden_dim),
            torch.zeros(1, 1, hidden_dim))

embed = torch.nn.Embedding(vocab_size, embedding_dim)
lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
linear = torch.nn.Linear(hidden_dim, vocab_size)
softmax = torch.nn.functional.softmax
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(embed.parameters()) + list(lstm.parameters()) + list(linear.parameters()), lr=0.001)

for x in range(num_epochs):
    print('Epoch: {}'.format(x))
    for encrypted, original in dataset:
        # encrypted.size() = [64]
        lstm_in = embed(encrypted)
        # lstm_in.size() = [64, 5]. This is a 2D tensor, but LSTM expects
        # a 3D tensor. So we insert a fake dimension.
        lstm_in = lstm_in.unsqueeze(1)
        # lstm_in.size() = [64, 1, 5]
        # Get outputs from the LSTM.
        lstm_out, lstm_hidden = lstm(lstm_in, zero_hidden())
        # lstm_out.size() = [64, 1, 10]
        # Apply the affine transform.
        scores = linear(lstm_out)
        # scores.size() = [64, 1, 27], but loss_fn expects a tensor
        # of size [64, 27, 1]. So we switch the second and third dimensions.
        scores = scores.transpose(1, 2)
        # original.size() = [64], but original should also be a 2D tensor
        # of size [64, 1]. So we insert a fake dimension.
        original = original.unsqueeze(1)
        # Calculate loss.
        loss = loss_fn(scores, original)
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()
    print('Loss: {:6.4f}'.format(loss.item()))

    with torch.no_grad():
        matches, total = 0, 0
        for encrypted, original in dataset:
            lstm_in = embed(encrypted)
            lstm_in = lstm_in.unsqueeze(1)
            lstm_out, lstm_hidden = lstm(lstm_in, zero_hidden())
            scores = linear(lstm_out)
            # Compute a softmax over the outputs
            predictions = softmax(scores, dim=2)
            # Choose the letter with the maximum probability
            _, batch_out = predictions.max(dim=2)
            # Remove fake dimension
            batch_out = batch_out.squeeze(1)
            # Calculate accuracy
            matches += torch.eq(batch_out, original).sum().item()
            total += torch.numel(batch_out)
        accuracy = matches / total
        print('Accuracy: {:4.2f}%'.format(accuracy * 100))
