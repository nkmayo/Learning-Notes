"""Something is off as the SiRNN doesn't learn, even with a 'tanh' activation which is essentially
just a regular RNN. Weight initialization can't be the 'sinusoid'?
Sources: 
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
https://www.youtube.com/watch?v=SEnXr6v2ifU
https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
https://vsitzmann.github.io/siren/
"""
# %% -------------Imports--------------------
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

"""
Okay, so the 'character RNN classification' tutorial actually uses a different RNN structure than
pytorch does internally. The tutorial combines (concatenates) the input and hidden layers into a 
longer vector before running a feed forward network to the output and hidden layers. Conversely,
it appears that PyTorch uses the approach outlined in the MIT lecture series (see sources) which is
to have the input and hidden layers feed forward (separately) and THEN combine (add together) to 
an form the new hidden layer, after the activation function is applied. This new hidden layer is
both fed forward to the output layer as well as copied over into the next iterations hidden input.

(Concatenated)
Mathematically, in the first case, you have c=cat(i_t,h_t-1) with size i.size + h.size. Then
h_t = W_hc * c 
o_t = activation(W_oc * c)
The two weight matricies, W_hc and W_oc, have a total number of elements
h*c + o*c = (h+o)*c = (h+o)*(h+i) = h*h + h*i + o*h + o*i

(Standard)
In the second case, you have
h_t = activation(W_hi * i + W_hh * h_t-1)
o_t = W_oh * h_t
The three weight matricies, W_hh, W_hi, and W_oh, have a total number of elements
h*h + h*i + o*h (o*i fewer)

Let's see if they have a similar 'per weight' response...
"""

# %% Standard Version
class SiRNN(nn.Module):
    def __init__(self, input_size, recurrent_size, output_size, bias=True,
                 is_first=False, omega_0=30, activation='sin'):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.activation = activation

        self.in2hid = nn.Linear(input_size, recurrent_size, bias=bias)
        self.hid2hid = nn.Linear(recurrent_size, recurrent_size, bias=False) # bias would be duplicate
        self.hid2out = nn.Linear(recurrent_size, output_size, bias=bias)
        
        self.init_weights()
    
    # so some appear to apply activation to the recurrent layer and pass the output, while others
    # apply activation to the output and pass the recurrent
    def forward(self, input, recurrent_previous):
        recurrent = self.in2hid(input) + self.hid2hid(recurrent_previous)
        if self.activation == 'tanh':
            recurrent = torch.tanh(recurrent)
        else:
            recurrent = torch.sin(self.omega_0 * recurrent)
        output = self.hid2out(recurrent)
        return output, recurrent

    # use the same initialization scheme from Vsitzmann et al.
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.in2hid.weight.uniform_(-1 / self.input_size, 
                                                1 / self.input_size)
                self.hid2hid.weight.uniform_(-1 / self.recurrent_size, 
                                                1 / self.recurrent_size)
                self.hid2out.weight.uniform_(-1 / self.recurrent_size, 
                                                1 / self.recurrent_size)       
            else:
                self.in2hid.weight.uniform_(-np.sqrt(6 / self.input_size) / self.omega_0, 
                                                np.sqrt(6 / self.input_size) / self.omega_0)
                self.hid2hid.weight.uniform_(-np.sqrt(6 / self.recurrent_size) / self.omega_0, 
                                                np.sqrt(6 / self.recurrent_size) / self.omega_0)
                self.hid2out.weight.uniform_(-np.sqrt(6 / self.recurrent_size) / self.omega_0, 
                                                np.sqrt(6 / self.recurrent_size) / self.omega_0)                                       

# Let's keep it simple to start and just have a single SiRNN followed by a dense layer and softmax output
class SiRNNet(nn.Module):
    def __init__(self, input_features, recurrent_size, intermediate_size, output_features, 
                 outermost_linear=True, first_omega_0=30, intermediate_omega_0=30.):
        super().__init__()
        self.recurrent_size = recurrent_size
        self.SiRNN = SiRNN(input_features, recurrent_size, intermediate_size,
                                  is_first=True, omega_0=first_omega_0)
        self.final_linear = nn.Linear(intermediate_size, output_features)
        self.softmax = nn.LogSoftmax(dim=1)

    
    def forward(self, input, hidden):
        # input = input.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output, new_hidden = self.SiRNN(input, hidden)
        output = self.final_linear(output)
        output = self.softmax(output)
        return output, new_hidden

    def zero_recurrent(self, batch_size):
        """Returns initial recurrent state: a tensor of zeros to initially input into the recurrent layer"""
        return torch.zeros(batch_size, self.recurrent_size) # pointless function?   

# %% Check to make sure dummy data works on both SiRNN and a SiRNNet
# model_S = SiRNN(50, 20, 10)
model = SiRNNet(50, 20, 15, 10)

loss_fn = nn.MSELoss()

batch_size = 10
TIMESTEPS = 5

# Create some fake data
batch = torch.randn(batch_size, 50)
hidden = torch.zeros(batch_size, 20)
target = torch.zeros(batch_size, 10)

loss = 0
for t in range(TIMESTEPS):
    # yes! you can reuse the same network several times,
    # sum up the losses, and call backward!
    output, hidden = model(batch, hidden)
    loss += loss_fn(output, target)
loss.backward()

# %% Concatenated version
class SiRNNc(nn.Module):
    def __init__(self, input_size, recurrent_size, output_size, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.recurrent_size = recurrent_size
        self.net_input = input_size + recurrent_size

        self.comb2hid = nn.Linear(input_size + recurrent_size, recurrent_size, bias=bias)
        self.comb2out = nn.Linear(input_size + recurrent_size, output_size, bias=bias)
        
        self.init_weights()
    
    # so some appear to apply activation to the recurrent layer and pass the output, while others
    # apply activation to the output and pass the recurrent
    def forward(self, x, recurrent_state):
        combined = torch.cat((x, recurrent_state), 1)
        recurrent = torch.sin(self.omega_0 * self.comb2hid(combined))
        output = self.comb2out(combined)
        return output, recurrent

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
    
    def init_recurrent(self, batch_size):
        """Returns initial recurrent state: a tensor of zeros to initially input into the recurrent layer"""
        return torch.zeros(batch_size, self.recurrent_size) # (batch_size, recurrent_size)

class SiRNNet2(nn.Module):
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
rnn = SiRNN(n_letters, n_hidden, n_categories, activation='tanh')

# %% The tutorials RNN that seems to work...
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
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))

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
# %% Something is off, it doesn't converge even when using 'tanh' activation
# or is it just the initial conditions with the sin activation?
criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    # for the SiRNN
    rnn.init_weights()
    hidden = torch.zeros(1, n_hidden)
    
    # initial hidden weights are the last self.hidden elements of init_weights
    # hidden = rnn.initHidden()

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
