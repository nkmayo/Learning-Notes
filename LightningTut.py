"""
Pytorch Lightning Tutorial
Compares building a simple ResNet image classifier with vanilla Pytorch and with Lightning
Source: https://www.youtube.com/watch?v=DbESHcCoWbM&list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2&index=3
"""
# %%
import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

# %% ---Conventional method---
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2+h1) # skip connection
        logits = self.l3(do)
        return logits

model = ResNet()

# Define my optimizer
params = model.parameters()
optimizer = optim.SGD(params, lr=1e-2)

# Define my loss
loss = nn.CrossEntropyLoss()

# Train, Val splits
train_data = datasets.MNIST('MNIST_data', 
                            train=True,
                            download=False,
                            transform=transforms.ToTensor()
                            )

train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)

# Define training and validation loops
nb_epochs = 5
for epoch in range(nb_epochs):
    losses = list()
    for batch in train_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        # 1 forward
        logits = model(x)
        
        # 2 compute the objective function
        J = loss(logits, y)

        # 3 cleaning the gradients
        model.zero_grad() # both model and optimizer contain parameters and can zero_grad
        # optimizer.zero_grad()
        # params.grad.zero_()

        # 4 accumulate the partial derivates of J wrt params
        J.backward()
        # params.grad.add_(dJ/dparams)

        # 5 Step in the direction opposite to gradient
        optimizer.step()
        # with torch.no_grad(): params = params - lr * params.grad

        losses.append(J.item())
    print(f'Epoch {epoch+1}, train loss: {torch.tensor(losses).mean():.2f}')

    losses = list()
    for batch in val_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        # 1 forward
        with torch.no_grad():
            logits = model(x)
        
        # 2 compute the objective function
        J = loss(logits, y)

        losses.append(J.item())

    print(f'Epoch {epoch+1}, val loss: {torch.tensor(losses).mean():.2f}')
# %% ---With Lightning---
import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
from torchmetrics.functional import accuracy

class ResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)

        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2+h1) # skip connection
        logits = self.l3(do)
        return logits
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2)
        # can have multiple optimizers
        # optimizer = optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        # 1 forward
        logits = self(x) # self is the model

        # 2 compute the objective function
        J = self.loss(logits, y)

        acc = accuracy(logits, y)
        pbar = {'train_acc': acc} # for some reason isn't displaying. Conflict with avg_val_acc?

        # return J
        return {'loss' : J, 'progress_bar': pbar} # equivalent statement
        # loss is reserved word, pl takes care of it
    
    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results['progress_bar']['val_acc'] = results['progress_bar']['train_acc']
        del results['progress_bar']['train_acc']
        return results

    def validation_epoch_end(self, val_step_outputs):
        # hooks
        # [results_batch_1, results_batch_2, ...]
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['val_acc'] for x in val_step_outputs]).mean()
        
        pbar = {'avg_val_acc': avg_val_acc}
        # certain reserved keywords, if not correct they won't be plotted
        return {'val_loss': avg_val_loss, 'progress_bar': pbar}

    def prepare_data(self):
        datasets.MNIST('MNIST_data', 
                            train=True,
                            download=False,
                            transform=transforms.ToTensor()
                            )

    def setup(self, stage = None): # throws error without specifying `stage` for some reason
        # In multi-GPU training, you're actually setting up a model per GPU
        train_data = datasets.MNIST('MNIST_data', 
                            train=True,
                            download=False,
                            transform=transforms.ToTensor()
                            )
        self.train_set, self.val_set = random_split(train_data, [55000, 5000])

    def train_dataloader(self):
        # if using multiple GPUs across clusters, you would create `prepare_data` and
        # `setup` definitions that handle things once (such as downloading dataset).
        # Dataloaders are not instantiated until you need them
        """Note: had to add `setup` as self.val_set was not found when this remained
        train_data = datasets.MNIST('MNIST_data', 
                            train=True,
                            download=False,
                            transform=transforms.ToTensor()
                            )

        self.train_set, self.val_set = random_split(train_data, [55000, 5000])
        # """
        train_loader = DataLoader(self.train_set, batch_size=32)
        return train_loader
 
    def val_dataloader(self):
        val_loader = DataLoader(self.val_set, batch_size=32)
        return val_loader   

model = ResNet()

trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=5) # don't want to crash by updating too often
trainer.fit(model)
## %% In terminal or notebook
# ! ls lightning_logs/checkpoints
# tensorboard --logdir ./lightning_logs
# %%
