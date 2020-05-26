import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int,
                 action_size: int,
                 seed: int = 42,
                 fully_connected_units_1: int = 64,
                 fully_connected_units_2: int = 64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fully_connected_1 (int): # Nodes in 1st FC hidden layer
            fully_connected_2 (int): # Nodes in 2nd FC hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fully_connected_1 = nn.Linear(state_size, 
                                           fully_connected_units_1)
        self.fully_connected_2 = nn.Linear(fully_connected_units_1, 
                                           fully_connected_units_2)
        self.fully_connected_3 = nn.Linear(fully_connected_units_2, 
                                           action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x_1 = self.fully_connected_1(state)
        x_relu_1 = F.relu(x_1)
        x_2 = self.fully_connected_2(x_relu_1)
        x_relu_2 = F.relu(x_2)
        action_values = self.fully_connected_3(x_relu_2)
        return action_values
