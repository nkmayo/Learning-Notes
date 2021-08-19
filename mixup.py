"""
Source:
https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/mixup
"""

# %% utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset


class MLPClassifierMixup(nn.Module):
    """Multilayer perceptron with inbuilt mixup logic.
    Assuming binary classification.
    Parameters
    ----------
    n_features : int
        Number of features.
    hidden_dims : tuple
        The sizes of the hidden layers.
    p : float
        Dropout probability.
    Attributes
    ----------
    hidden_layers : nn.ModuleList
        List of hidden layers that are each composed of a `Linear`,
        `LeakyReLU` and `Dropout` modules.
    n_hidden : int
        Number of hidden layers.
    clf : nn.Linear
        The classifier at the end of the pipeline.
    """

    def __init__(self, n_features, hidden_dims, p=0):
        super().__init__()
        dims = (n_features,) + hidden_dims

        self.n_hidden = len(hidden_dims)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p),
                )
                for i in range(self.n_hidden)
            ]
        )
        self.clf = nn.Linear(dims[-1], 1) # classifier input

    def forward(self, x, start=0, end=None):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Input of shape `(n_samples, dim)`. Note that the dim
            will depend on `start`.
        start : int
            The hidden layer where the forward pass starts (inclusive). We
            use a convention of `start=0` and `end=0` as a noop and the input
            tensor is returned. Useful for implementing input mixing.
        end : int or None
            The ending hidden layer (exclusive). If None, then always run until
            the last hidden layer and then we also apply the classifier.
        """
        for module in self.hidden_layers[start:end]:
            x = module(x)

        if end is None:
            x = self.clf(x)

        return x


class CustomDataset(Dataset):
    """Custom classification dataset assuming we have X and y loaded in memory.
    Parameters
    ----------
    X : np.ndarray
        Features of shape `(n_samples, n_features)`.
    y : np.ndarray
        Targets of shape `(n_samples,)`.
    """

    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError("Inconsistent number of samples")

        classes = np.unique(y)
        if not np.array_equal(np.sort(classes), np.array([0, 1])):
            raise ValueError

        self.X = X
        self.y = y

    def __len__(self):
        """Compute the length of the dataset."""
        return len(self.X)

    def __getitem__(self, ix):
        """Return a single sample."""
        return self.X[ix], self.y[ix]


def generate_spirals(
    n_samples,
    noise_std=0.05,
    n_cycles=2,
    random_state=None,
):
    """Generate two spirals dataset.
    Parameters
    ----------
    n_samples : int
        Number of samples to generate. For simplicity, an even number
        is required. The targets (2 spirals) are perfectly balanced.
    noise_std : float
        Standard deviation of the noise added to the spirals.
    n_cycles : int
        Number of revolutions the spirals make.
    random_state : int or None
        Controls randomness.
    Returns
    -------
    X : np.ndarray
        Features of shape `(n_samples, n_features)`.
    y : np.ndarray
        Targets of shape `(n_samples,)`. There are two
        classes 0 and 1 representing the two spirals.
    """
    if n_samples % 2 != 0:
        raise ValueError("The number of samples needs to be even")

    n_samples_per_class = int(n_samples // 2)

    angle_1 = np.linspace(0, n_cycles * 2 * np.pi, n_samples_per_class)
    angle_2 = np.pi + angle_1
    radius = np.linspace(0.2, 2, n_samples_per_class)

    x_1 = radius * np.cos(angle_1)
    y_1 = radius * np.sin(angle_1)

    x_2 = radius * np.cos(angle_2)
    y_2 = radius * np.sin(angle_2)

    X = np.concatenate(
        [
            np.stack([x_1, y_1], axis=1),
            np.stack([x_2, y_2], axis=1),
        ],
        axis=0,
    )
    y = np.zeros((n_samples,))
    y[n_samples_per_class:] = 1.0

    if random_state is not None:
        np.random.seed(random_state)

    new_ixs = np.random.permutation(n_samples)

    X = X[new_ixs] + np.random.normal(
        loc=0, scale=noise_std, size=(n_samples, 2)
    )
    y = y[new_ixs]

    return X, y


def generate_prediction_img(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Generate contour and scatter plots with predictions.
    Parameters
    ----------
    model : MLPClassifierMixup
        Instance of a multilayer-perceptron.
    X_train, X_test : np.ndarray
        Trand and test features of shape `(n_samples, n_features)`.
    y_train, y_test : np.ndarray
        Train and test targets of shape `(n_samples,)`.
    Yields
    ------
    matplotlib.Figure
        Different figures.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    delta = 0.5

    xlim = (X_test[:, 0].min() - delta, X_test[:, 0].max() + delta)
    ylim = (X_test[:, 1].min() - delta, X_test[:, 1].max() + delta)

    n = 50
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], n),
        np.linspace(ylim[0], ylim[1], n),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    with torch.no_grad():
        logits = model(torch.from_numpy(grid).to(device, dtype))

    probs = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()

    probs = probs.reshape(xx.shape)

    fig, ax = plt.subplots(1, 1, dpi=170)

    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k"
    )
    ax.set_title("Test data")

    yield fig
    ax.cla()

    ax.contourf(xx, yy, probs, cmap=cm, alpha=0.8)
    ax.set_title("Prediction contours")

    yield fig

    ax.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
    )
    ax.set_title("Train data + prediction contours")

    yield fig

# %% train.py
"""Makes a CLI to easily run training loops with various settings"""

import argparse
import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def main(argv=None):
    parser = argparse.ArgumentParser("Training")

    # Parameters
    parser.add_argument(
        "logpath",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--mixup",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--dropout-probability",
        type=float,
        default=0,
        help="The probability of dropout",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=(32, 32, 32),
        help="Hidden dimensions of the MLP",
    )
    parser.add_argument(
        "-c",
        "--n-cycles",
        type=float,
        default=2,
        help="Number of cycles when creating the spiral dataset",
    )
    parser.add_argument(
        "-n",
        "--n-epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        "-k",
        "--mixing-layer",
        type=int,
        nargs=2,
        default=(None, None),
        help="The range of k to sample from",
    )
    parser.add_argument(
        "-s",
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples",
    )
    parser.add_argument(
        "-r",
        "--random-state",
        type=int,
        default=5,
        help="Random state",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay",
    )

    args = parser.parse_args(argv)

    device = torch.device("cpu")
    dtype = torch.float32

    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    # Dataset preparation
    X, y = generate_spirals(
        args.n_samples,
        noise_std=0,
        n_cycles=args.n_cycles,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.9,
        shuffle=True,
        stratify=y,
    )

    X_test_t = torch.from_numpy(X_test).to(device, dtype)

    dataset_train = CustomDataset(X_train, y_train)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=2 * args.batch_size,
        drop_last=True,
        shuffle=True,
    )

    # Model and loss definition
    model = MLPClassifierMixup(
        n_features=2,
        hidden_dims=tuple(args.hidden_dims),
        p=args.dropout_probability,
    )
    model.to(device, dtype)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=args.weight_decay,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Summary
    writer = SummaryWriter(args.logpath)
    writer.add_text("hparams", json.dumps(vars(args)))

    # Training + evaluation loop
    bs = args.batch_size
    n_steps = 0
    for e in range(args.n_epochs):
        for X_batch, y_batch in dataloader_train:
            X_batch, y_batch = X_batch.to(device, dtype), y_batch.to(
                device, dtype
            )
            if args.mixup:
                k_min, k_max = args.mixing_layer
                k_min = k_min or 0
                k_max = k_max or model.n_hidden + 1

                k = np.random.randint(k_min, k_max)
                lam = np.random.beta(2, 2)
                writer.add_scalar("k", k, n_steps)
                writer.add_scalar("lambda", lam, n_steps)

                h = model(X_batch, start=0, end=k)  # (2 * batch_size, *)

                h_mixed = lam * h[:bs] + (1 - lam) * h[bs:]  # (batch_size, *)
                y_mixed = lam * y_batch[:bs] + (1 - lam) * y_batch[bs:]  # (batch_size,)

                logits = model(h_mixed, start=k, end=None)  # (batch_size, 1)
                loss = loss_fn(logits.squeeze(), y_mixed)

            else:
                logits = model(X_batch[:bs])  # (batch_size, 1)
                loss = loss_fn(logits.squeeze(), y_batch[:bs])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            writer.add_scalar("loss_train", loss, n_steps)

            if n_steps % 2500 == 0:
                model.eval()
                fig_gen = generate_prediction_img(
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                )
                writer.add_figure("test", next(fig_gen))
                writer.add_figure("contour", next(fig_gen), n_steps)
                writer.add_figure("contour_train", next(fig_gen), n_steps)

                with torch.no_grad():
                    logits_test = model(X_test_t).squeeze().detach().cpu()

                acc_test = (
                    torch.sigmoid(logits_test).round().numpy() == y_test
                ).sum() / len(y_test)
                loss_test = loss_fn(logits_test, torch.from_numpy(y_test))

                writer.add_scalar("loss_test", loss_test, n_steps)
                writer.add_scalar("accuracy_test", acc_test, n_steps)

                model.train()

            n_steps += 1


if __name__ == "__main__":
    main()

# %% launch_experiments
"""
# for me
python Learning-Notes/mixup.py -r 123 -n 100000 -s 1000 tb_results/123 --mixup

# launch with
tensorboard --logdir=.\tb_results\123
# and view at http://localhost:6006/

# his bash script
set -x # in bash set -x enables a mode of the shell where all executed commands are printed to the terminal.

N_EPOCHS=100000
N_SAMPLES=1000
SEED=123
TBOARD_DIR=tb_results/$SEED

python mixup.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/no_regularization
python mixup.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/weight_decay --weight-decay 0.6
python mixup.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/dropout -p 0.2 
python mixup.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/mixup --mixup 
python mixup.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/input_mixup -k 0 1 --mixup
python mixup.py -r $SEED -n $N_EPOCHS -s $N_SAMPLES $TBOARD_DIR/hidden_layers_mixup -k 1 4 --mixup
"""
# %%
