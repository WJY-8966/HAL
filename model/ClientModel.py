import torch
from torch import nn


class BottomModel(nn.Module):
    def __init__(self, n_f_in, n_f_out):
        super().__init__()
        self.dense = nn.Linear(n_f_in, n_f_out)
        nn.init.xavier_normal_(self.dense.weight)

    def forward(self, x):
        x = self.dense(x)
        x = torch.relu(x)
        return x


class TopModel(nn.Module):
    def __init__(self, input_size, num_labels):
        super().__init__()
        self.dense_1 = nn.Linear(input_size, 512, bias=True)
        nn.init.xavier_normal_(self.dense_1.weight)
        self.dense_2 = nn.Linear(512, num_labels, bias=True)
        nn.init.xavier_normal_(self.dense_2.weight)

    def forward(self, x):
        x = self.dense_1(x)
        x = torch.relu(x)
        x = self.dense_2(x)
        x = torch.relu(x)

        return x