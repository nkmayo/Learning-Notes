"""
Pytorch Lightning Tutorial revamping a variational autoencoder

Source: https://www.youtube.com/watch?v=QHww1JH7IDU
https://github.com/pytorch/examples/blob/master/vae/main.py
"""
# %%
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import datetime
import pytorch_lightning as pl
from PIL import Image
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% Regular Encoder/Decoder
class VAE(pl.LightningModule):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, z):
        return self.decode(z)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
    
    def save_image(self, data, filename):
        img = data.clone().clamp(0, 255).numpy()
        img = img[0] # just taking first image here
        img = Image.fromarray(img, mode='RGB')
        img.save(filename)
        
    def training_step(self, batch, batch_idx):
        x, _ = batch

        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)

        loss = self.loss_function(x_hat, x, mu, logvar)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)

        val_loss = self.loss_function(x_hat, x, mu, logvar)
        
        return {'val_loss': val_loss, 'x_hat': x_hat}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('MNIST_data', train=True, download=False,
            transform=transforms.ToTensor()), num_workers=4,
            batch_size=args.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('MNIST_data', train=False, download=False,
            transform=transforms.ToTensor()), num_workers=4,
            batch_size=args.batch_size, shuffle=False)
        return val_loader
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        x_hat = outputs[-1]['x_hat']

        # for some reason doesn't appear to be exporting images
        grid = torchvision.utils.make_grid(x_hat)
        self.logger.experiment.add_image('images', grid, 0)

        log = {'avg_val_loss': val_loss}
        return {'log': log}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='VAE MNIST Example')
    # parser = pl.Trainer.add_argparse_args(parser) # adds trainer args to enable defaults, but conflicts with ipykernel
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args, _ = parser.parse_known_args() # parser.parse_args() conflicts with ipykernel

    vae = VAE()
    # trainer = pl.Trainer.from_argparse_args(args,fast_dev_run=True)
    trainer = pl.Trainer(fast_dev_run=True)
    # can insert fast_dev_run=True for quick check
    # train_percent_check=0.1, val_percent_check=0.1 (apparently aren't recognized?)
    trainer.fit(vae)
# %%
