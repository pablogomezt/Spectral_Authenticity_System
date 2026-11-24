
import torch 

def lossimage_2(y_true, y_pred):
    a_b = torch.sum(y_pred * y_true, dim=-1)
    mag_a = torch.sqrt(torch.sum(y_pred ** 2, dim=-1))
    mag_b = torch.sqrt(torch.sum(y_true ** 2, dim=-1))
    spectral_loss = torch.mean((a_b / (mag_a * mag_b)) ** 2 - 1)
    return spectral_loss


def total_variation_in_third_dimension(img):
    return torch.mean((img[:,  1:] - img[:, :-1]) ** 2)