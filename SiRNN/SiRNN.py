"""This SiRNN technically runs, but doesn't currently learn and gets stuck always predicting the 
same value in the name recognition test. The SiRNNet optionally adds a dense layer at the end and
includes a softmax function for classification, but needs to be fixed, as the hidden layers are 
not appropriately passed (or initiated). Need to manually define the forward method rather than
calling Sequential.
Sources: 
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
https://vsitzmann.github.io/siren/
"""
# %% -------------Imports--------------------
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

# %%
# maybe later allow choice of activations (for easy comparison)
class SiRNN(nn.Module):
    # 
    def __init__(self, input_size, hidden_size, output_size, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.hidden_size = hidden_size
        self.net_input = input_size + hidden_size

        self.comb2hid = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.comb2out = nn.Linear(input_size + hidden_size, output_size, bias=bias)
        
        self.init_weights()
    
    # so some appear to apply activation to the hidden layer and pass the output, while others
    # apply activation to the output and pass the hidden
    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.sin(self.omega_0 * self.comb2hid(combined))
        output = self.comb2out(combined)
        return output, hidden

    # use the same initialization scheme from Vsitzmann et al.
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.comb2hid.weight.uniform_(-1 / self.net_input, 
                                                1 / self.net_input)
                self.comb2out.weight.uniform_(-1 / self.net_input, 
                                                1 / self.net_input)       
            else:
                self.comb2hid.weight.uniform_(-np.sqrt(6 / self.net_input) / self.omega_0, 
                                                np.sqrt(6 / self.net_input) / self.omega_0)
                self.comb2out.weight.uniform_(-np.sqrt(6 / self.net_input) / self.omega_0, 
                                                np.sqrt(6 / self.net_input) / self.omega_0)
    
    def initHidden(self):
        """Returns initial hidden state: a tensor of zeros to initially input into the hidden layer"""
        return torch.zeros(1, self.hidden_size) # (batch_size, hidden_size)

class SiRNNet(nn.Module):
    def __init__(self, in_features, hidden_size, intermediate_size, hidden_layers, out_features, 
                 outermost_linear=False, first_omega_0=30, intermediate_omega_0=30.):
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.net = []
        self.net.append(SiRNN(in_features, hidden_size, intermediate_size,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SiRNN(intermediate_size, hidden_size, intermediate_size,
                                      is_first=False, omega_0=intermediate_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(intermediate_size, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / intermediate_size) / intermediate_omega_0, 
                                              np.sqrt(6 / intermediate_size) / intermediate_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SiRNN(intermediate_size, hidden_size, out_features, 
                                      is_first=False, omega_0=intermediate_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        output = self.softmax(output) # added softmax
        return output, coords        

# %%
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
# %%
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())

n_hidden=128
rnn = SiRNN(n_letters, n_hidden, n_categories)
# %%
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))
# %%
import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
# %% Need to use a SiRNNet not just a SiRNN as we need to softmax the output to make sense of it
criterion = nn.NLLLoss()

learning_rate = 0.05 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    rnn.init_weights()
    # initial hidden weights are the last self.hidden elements of init_weights
    # hidden = rnn.in2hidden.weight
    hidden = torch.zeros(1, n_hidden)
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
# %% A 'conventional' RNN (though only with a softmax activation...)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# %%
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    rnn.init_weights()
    # initial hidden weights are the last self.hidden elements of init_weights
    # hidden = rnn.in2hidden.weight
    hidden = torch.zeros(1, n_hidden)

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
# %%
