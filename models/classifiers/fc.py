import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


class FC(nn.Module):
    def __init__(self, num_bands, num_classes, *args, **kwargs):
        super().__init__()
        self.name = 'fc'

        self.fc1 = nn.Linear(num_bands, 128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.act3 = nn.ReLU()
        self.out = nn.Linear(32, num_classes)

        self.apply(init_weights)

    def apply_l2_regularization(self, reg_param=None):
        loss = 0
        if reg_param is not None:
            for module in self.modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    loss += reg_param * torch.norm(module.weight, p=2)
                    loss += reg_param * torch.norm(module.bias, p=2)

        return loss

    def adjust_input(self, x):
        if x.ndim in [1, 2, 3]:
            x = x.squeeze()
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        return x

    def forward(self, x):
        x = self.adjust_input(x)
        x1 = self.act1(self.fc1(x))
        x2 = self.act2(self.fc2(x1))
        x3 = self.act3(self.fc3(x2))
        x4 = self.out(x3)

        return x4
