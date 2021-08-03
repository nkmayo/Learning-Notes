"""Example of using forward hooks

Sources
-------
https://www.youtube.com/watch?v=1ZbLA7ofasY&t=961s
https://github.com/jankrepl/mildlyoverfitted/tree/master/mini_tutorials/visualizing_activations_with_forward_hooks
"""
# %% -------------IMPORTS--------------
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.utils.tensorboard import SummaryWriter
import pathlib

# %% ---BUILD NETWORK AND APPLY HOOKS---
class Network(Module):
    def __init__(self):
        super().__init__()

        self.fc_1 = Linear(10, 20)
        self.fc_2 = Linear(20, 40)  # match previous output layer
        self.fc_3 = Linear(40, 2)   # match previous output layer

    def forward(self, x):
        # forward method should only contain info related to the forward pass
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)

        x = F.relu(x)

        return x

if __name__ == "__main__":
    # instantiate the writer
    log_dir = pathlib.Path.cwd() / 'tensorboard_logs' # cwd = current working directory
    writer = SummaryWriter(log_dir)
    
    x = torch.rand(1, 10)   # create random input vector
    net = Network() # instantiate Network

    # define the activation hook
    # hook(module, input, output) format required even though inp isn't used here
    def activation_hook(inst, inp, out):
        """Run activation hook
        
        Parameters
        ----------
        inst: torch.nn.Module
            The layer we want to attach the hook to
        inp: torch.Tensor
            The input to the `forward` method.
        out: torch.Tensor
            The output of the `forward` method
        """
        print('Here')

        writer.add_histogram(repr(inst), out) # see tensorboard docs

    # net has no hooks assigned to it when first instantiated
    # forward hooks need to be registered before running forward pass
    net.fc_1.register_forward_hook(activation_hook)
    net.fc_2.register_forward_hook(activation_hook)
    net.fc_3.register_forward_hook(activation_hook)

    # a Module's `forward` method is run automatically by `__call__` when an input is passed
    y = net(x)

    print(y)
    # to see results, call (and navigate to 'HISTOGRAMS'):
    # tensorboard --logdir=tensorboard_logs/
# %%
