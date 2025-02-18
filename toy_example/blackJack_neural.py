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

    def __init__(self, device):
        super(neural_agent, self).__init__()
        self.decision_boundary = nn.Parameter(
            torch.tensor([15.0 for _ in range(20)], device=device)
        )

    def forward(self, input_state):
        """
        input_state: 4 states representing each case
        """
        (mask, total_val) = input_state[0], input_state[1]
        decision = self.decision_boundary - total_val.view(-1, 1).expand(-1, 20)
        output = mask * decision

        return torch.sigmoid(5 * torch.sum(output, dim=1))
