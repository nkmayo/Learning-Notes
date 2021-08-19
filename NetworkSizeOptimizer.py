"""
There seems to be a generic problem with neural networks in that we are often unsure as to what 
the optimal number and sizes of the hidden layers should be. For example, autoencoders often 
strive to achieve a minimum central hidden layer size retaining all of the "hidden features" 
necessary to accurately recreate the original input. It can be difficult to converge to the 
global minima using gradient descent as overfitting with complex networks leads to getting stuck
in local minima. The idea here is to start with a network that has too few parameters and
underfits the data and then alternately train and grow the complexity of the network. The hope
is that the smoother underfit boundaries begin to increase in curvature as the network grows,
thereby providing more assurance that the minimum converged to is the global minimum rather than
a local minimum. Furthermore, the hope is that by starting small and growing the network you
end up saving time by not training unnecessarily large networks.
Let's use a 'simple' feed forward network and the MNIST dataset as an example to practice on...
"""
# %%
import torch
from torch._C import device
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import datetime
from matplotlib import pyplot as plt

# %% Define the network
class myAutoenc(nn.Module):
    """Multilayer perceptron with inbuilt mixup logic.
    Assuming binary classification.
    Parameters
    ----------
    n_features : int
        Number of features.
    hidden_dims : tuple
        The sizes of the hidden layers.
    convChan : int
        Number of convolutional channels output to hidden layers.
    Attributes
    ----------
    n_input : int
        Flattened input dimension size.
    n_hidden : int
        Number of hidden layers.
    input : nn.Conv2d
        Initial 2D convolutional layer
    flatten : nn.Flatten
        Layer to flatten the convolutional output
    lin1 : nn.Linear
        Linear layer that connects the flattened convolutional output
        with the subsequent hidden layers
    hidden_layers : nn.ModuleList
        List of hidden layers that are each composed of a `Linear`
        and `LeakyReLU` module.
    output : nn.Linear
        Output layer at the end of the autoencoder. Ideally matches
        the input once reshaped to the input feature dimensions.

    """

    def __init__(self, n_features=28*28, hidden_dims=(4,4), convChan=5):
        super().__init__()
        self.n_input = n_features
        self.n_hidden = len(hidden_dims)
        self.input = nn.Conv2d(1, convChan, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(n_features*convChan, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2),
                )
                for i in range(self.n_hidden-1)
            ]
        )
        self.output = nn.Linear(hidden_dims[-1], n_features) # output same as input size

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Input of shape `(1, n_features)`.
        """
        x = self.input(x)
        x = self.flatten(x)
        x = self.lin1(x)

        for module in self.hidden_layers:
            x = module(x)

        x = self.output(x)

        return x # x.view(-1,28,28)

# %% Construct the training loop
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0

        for imgs, _ in train_loader:
            imgs = imgs.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs.view(1,-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print(
                "{} Epoch {}, Training loss {}".format(
                    datetime.datetime.now(),
                    epoch,
                    loss_train / len(train_loader),
                )
            )

# %% Instantiate/Define Model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training on device {device}.")


data_path = "MNIST_data/"
mnist = datasets.MNIST(
    data_path,
    train=False,
    download=True,
    transform=transforms.ToTensor() # convert from PIL to (1,28,28) tensor
    )

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, shuffle=True)

model = myAutoenc().to(device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

numel_list = [p.numel() for p in model.parameters()]
print(sum(numel_list), numel_list)

training_loop(
    n_epochs=100,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader,
)
# %%

# %% Look at the data...at the moment it's just highlighting entire regions where the
# numbers typically appear
inimg, _ = mnist[0]
plt.imshow(inimg.permute(1,2,0)) # have to switch from CxHxW to HxWxC before plotting

with torch.no_grad(): # don't want the gradients being updated just to view an output
    outimg = model(inimg.unsqueeze(0)).view(-1,28,28)

plt.imshow(outimg.permute(1,2,0))
# %% Before training example (random output)
untrained = myAutoenc()
with torch.no_grad(): # don't want the gradients being updated just to view an output
    untrainedOut = untrained(inimg.unsqueeze(0)).view(-1,28,28)

plt.imshow(untrainedOut.permute(1,2,0))
# %%
