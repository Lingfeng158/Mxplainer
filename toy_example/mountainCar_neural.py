# Model part
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
from itertools import chain


class neural_agent(nn.Module):
    """
    Computation for opponent_desire_prob, held_prob, discard_prob
    """

    def __init__(self):
        super(neural_agent, self).__init__()
        self.paramA = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.paramB = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.paramC = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.paramD = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, input_state):
        """
        input_state: 4 states representing each case
        """
        return (
            input_state[:, 0] * self.paramA
            + input_state[:, 1] * self.paramB
            + input_state[:, 2] * self.paramC
            + input_state[:, 3] * self.paramD
        )
