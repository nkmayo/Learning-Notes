"""Source:
https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html#:~:text=Internally%2C%20PyTorch%20uses%20a%20Collate%20Function%20to%20combine,we%E2%80%99re%20going%20to%20assume%20automatic%20batching%20is%20enabled
"""
# %%
import torch

# create a dummy dataset
xs = list(range(10))
ys = list(range(10,20))

class MyDataset:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
    
    def __getitem__(self, i):
        return self.xs[i], self.ys[i]
    
    def __len__(self):
        return len(self.xs)

dataset = MyDataset(xs, ys)

# create a custom batch sampler
from torch.utils.data import Sampler
import random

def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)

class EachHalfTogetherBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        halfway_point = len(dataset) // 2 
        self.first_half_indices = list(range(halfway_point))
        self.second_half_indices = list(range(halfway_point, len(dataset)))
        self.batch_size = batch_size
    
    def __iter__(self):
        random.shuffle(self.first_half_indices)
        random.shuffle(self.second_half_indices)
        first_half_batches  = chunk(self.first_half_indices, self.batch_size)
        second_half_batches = chunk(self.second_half_indices, self.batch_size)
        combined = list(first_half_batches + second_half_batches)
        combined = [batch.tolist() for batch in combined]
        random.shuffle(combined)
        return iter(combined)
    
    def __len__(self):
        return (len(self.first_half_indices) + len(self.second_half_indices)) // self.batch_size

# create an instance of the dataloader using the custom batch sampler
batch_size = 3
each_half_together_batch_sampler = EachHalfTogetherBatchSampler(dataset, batch_size)
for x in each_half_together_batch_sampler:
    print(x)

from torch.utils.data import DataLoader
for i, (xb,yb) in enumerate(DataLoader(dataset, batch_sampler=each_half_together_batch_sampler)):
    print(f'Batch #{i}. x{i}:', xb)
    print(f'          y{i}:', yb)
# %% Using a custom collate function
# One thing that custom collate functions are often used for is for padding variable length batches.
# So let's change our dataset so that each x is a list, and they're all different sizes.

xs = list([torch.randint(1, 10, (x,)) for x in range(1, 11)])
print('xs starting:')
for _, x in enumerate(xs):
    print(x)
dataset = list(zip(xs,ys))

# Now, if we try with the default collate function, it'll raise a RuntimeError.
"""
try:
    for xb, yb in DataLoader(dataset, batch_size=2):
        print(xb)
except RuntimeError as e:
    print('RuntimeError: ', e)
"""
# pad the xs to match the longest in the batch

from torch.nn.utils.rnn import pad_sequence

def pad_x_collate_function(batch):
    # batch looks like [(x0,y0), (x4,y4), (x2,y2)... ]
    xs = [sample[0] for sample in batch]
    ys = [sample[1] for sample in batch] 
    # If you want to be a little fancy, you can do the above in one line 
    # xs, ys = zip(*samples) 
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    return xs, torch.tensor(ys)

print('\nxs after paddinging (batch size = 3):')
for xb, yb in DataLoader(dataset, batch_size=3, collate_fn=pad_x_collate_function):
    print(xb)

print('\nWith shuffling=True')
for xb, yb in DataLoader(dataset, shuffle=True, batch_size=3, collate_fn=pad_x_collate_function):
    print('xs: ', xb)
    print('ys: ', yb)

# But there's a bit of an issue, some of the smaller values look the they have too much padding.
# Luckily, we've already created something that'll help here. We can use our EachHalfTogetherBatchSampler 
# custom batch_sampler so that the first and second half are batched separately.

print('\nCustom batch sampler combined with custom collate function')
each_half_together_batch_sampler = EachHalfTogetherBatchSampler(dataset, batch_size=2)
for xb, yb in DataLoader(dataset, collate_fn=pad_x_collate_function, batch_sampler=each_half_together_batch_sampler):
    print('xs: ', xb)
    print('ys: ', yb)
# %%
