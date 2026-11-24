import torch
import torch.nn as nn
from einops import rearrange


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


class CNN(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()
        self.name = 'cnn'

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1), stride=(2, 1))
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(2, 1))
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1))
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1))
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1))
        self.act5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3, 1), stride=(2, 1))
        self.act6 = nn.ReLU()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1024, 128)
        self.act7 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.act8 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.act9 = nn.ReLU()
        self.fc4 = nn.Linear(32, num_classes)

        self.apply(init_weights)

    def apply_l2_regularization(self, reg_param=1e-4):
        loss = 0
        if reg_param is not None:
            modules = [self.fc1, self.fc2, self.fc3]

            for module in modules:
                loss += reg_param * torch.norm(module.weight, p=2)
                loss += reg_param * torch.norm(module.bias, p=2)

        return loss

    def adjust_input(self, x):
        if x.ndim == 2:
            x = x[:, None, :, None]
        if x.ndim == 3:
            x = x[:, None, ...]
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        return x

    def forward(self, x):
        x = self.adjust_input(x)
        x1 = self.act1(self.conv1(x))
        x2 = self.act2(self.conv2(x1))
        x3 = self.act3(self.conv3(x2))
        x4 = self.act4(self.conv4(x3))
        x5 = self.act5(self.conv5(x4))
        x6 = self.act6(self.conv6(x5))

        x_flat = self.flat(x6)
        x7 = self.act7(self.fc1(x_flat))
        x8 = self.act8(self.fc2(x7))
        x9 = self.act9(self.fc3(x8))
        x10 = self.fc4(x9)

        return x10
