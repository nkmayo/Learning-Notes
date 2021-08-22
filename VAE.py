"""
Variational autoencoder using the MNIST data set

Can we play around with the loss function to see how that affects blur?

Source: https://avandekleut.github.io/vae/
"""
# %%
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% Regular Encoder/Decoder
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))

data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('MNIST_data',
               transform=torchvision.transforms.ToTensor(),
               download=False),
        batch_size=128,
        shuffle=True)

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

# %% Variational Encoder/Decoder
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x)) # could also use F.softmax(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape) # adding gaussian noise to the latent space
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    alpha = 0.001 # added hyperparameter to minimize 'blur', does it work?
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            l1 = ((x - x_hat)**2).sum()
            kl = autoencoder.encoder.kl
            #loss = l1 + kl # usual loss definition
            # let's add coupling between the two loss terms so that it compromises
            # between error and being too far from a standard normal distribution
            loss = (l1 + kl) + alpha*l1*kl # only seems to reduce clustering
            loss.backward()
            opt.step()
        
        if epoch == epochs or epoch % 3 == 0:
            print(
                "{} Epoch {}, l1 {}, KL {}, Loss {}".format(
                    datetime.datetime.now(),
                    epoch,
                    l1,
                    kl,
                    loss
                    )
                )
    return autoencoder

latent_dims = 2

vae = VariationalAutoencoder(latent_dims).to(device) # GPU
vae = train(vae, data)

plot_latent(vae, data) 

# %% only works with latent_dims=2
#plot_reconstructed(vae)
plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
# %%