import os
from pathlib import Path
import yaml

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from models.classifiers.fc import FC
from models.classifiers.cnn import CNN
from models.classifiers.spectralformer import SpectralFormer
import numpy as np
import torch
import matplotlib
# matplotlib.use('wxAgg')

def load_yaml(config):
    with open(config, 'r') as file:
        return yaml.safe_load(file)


def save_yaml(path, config):
    with open(f'{path}/config.yaml', 'w') as file:
        yaml.dump(config, file)


def experiment_setting(script_name, save_dir, args):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    save_config(save_dir, os.path.basename(script_name), args)  # Save the experiment config in a .txt file

def lossimage_2(y_true, y_pred):
    a_b = torch.sum(y_pred * y_true, dim=-1)
    mag_a = torch.sqrt(torch.sum(y_pred ** 2, dim=-1))
    mag_b = torch.sqrt(torch.sum(y_true ** 2, dim=-1))
    spectral_loss = torch.mean((a_b / (mag_a * mag_b)) ** 2 - 1)
    return spectral_loss



def save_config(save_path, file, args):
    txt_name = '/model_info.txt'

    with open(save_path + txt_name, 'w') as txt:
        txt.write('#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\n')
        txt.write('             Model information             \n')
        txt.write('#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\n')
        txt.write(f'python {file} \\\n')
        for arg in vars(args):
            name = arg.replace('_', '-')
            value = getattr(args, arg)

            if isinstance(value, dict):
                value = f'"{value}"'
            txt.write(f'       --{name} {value} \\\n')


CLASSIFIER_NAME = {
    'fc': FC,
    'cnn': CNN,
    'sf': SpectralFormer
}

REGULARIZER_NAME = {
    'none': None
}

SCHEDULER_NAME = {
    'none': None,
    'step_lr': StepLR,
    'reduce_lr': ReduceLROnPlateau
}


def get_classifier(class_config):
    classifier = CLASSIFIER_NAME[class_config['name'].lower()]
    return classifier(**class_config['params'])


def get_scheduler(name):
    return SCHEDULER_NAME[name.lower()]




def g(x, alpha, mu, sigma1, sigma2):
    sigma = (x < mu) * sigma1 + (x >= mu) * sigma2
    return alpha * np.exp((x - mu) ** 2 / (-2 * (sigma ** 2)))


def component_x(x): return g(x, 1.056, 5998, 379, 310) + \
    g(x, 0.362, 4420, 160, 267) + g(x, -0.065, 5011, 204, 262)


def component_y(x): return g(x, 0.821, 5688, 469, 405) + \
    g(x, 0.286, 5309, 163, 311)


def component_z(x): return g(x, 1.217, 4370, 118, 360) + \
    g(x, 0.681, 4590, 260, 138)


def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1 - x - y))


ILUMINANT = {
    'D65': xyz_from_xy(0.3127, 0.3291),
    'E': xyz_from_xy(1 / 3, 1 / 3),
}

COLOR_SPACE = {
    'sRGB': (xyz_from_xy(0.64, 0.33),
             xyz_from_xy(0.30, 0.60),
             xyz_from_xy(0.15, 0.06),
             ILUMINANT['D65']),

    'AdobeRGB': (xyz_from_xy(0.64, 0.33),
                 xyz_from_xy(0.21, 0.71),
                 xyz_from_xy(0.15, 0.06),
                 ILUMINANT['D65']),

    'AppleRGB': (xyz_from_xy(0.625, 0.34),
                 xyz_from_xy(0.28, 0.595),
                 xyz_from_xy(0.155, 0.07),
                 ILUMINANT['D65']),

    'UHDTV': (xyz_from_xy(0.708, 0.292),
              xyz_from_xy(0.170, 0.797),
              xyz_from_xy(0.131, 0.046),
              ILUMINANT['D65']),

    'CIERGB': (xyz_from_xy(0.7347, 0.2653),
               xyz_from_xy(0.2738, 0.7174),
               xyz_from_xy(0.1666, 0.0089),
               ILUMINANT['E']),
}


class ColourSystem:

    def __init__(self, start=380, end=750, num=100, cs='sRGB'):
        # Chromaticities
        bands = np.linspace(start=start, stop=end, num=num) * 10

        self.cmf = np.array([component_x(bands),
                             component_y(bands),
                             component_z(bands)])

        self.red, self.green, self.blue, self.white = COLOR_SPACE[cs]

        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)

        # White scaling array
        self.wscale = self.MI.dot(self.white)

        # xyz -> rgb transformation matrix
        self.A = self.MI / self.wscale[:, np.newaxis]

    def get_transform_matrix(self):
        XYZ = self.cmf
        RGB = XYZ.T @ self.A.T
        RGB = RGB / np.sum(RGB, axis=0, keepdims=True)
        return RGB

    def spec_to_rgb(self, spec):
        """Convert a spectrum to an rgb value."""
        M = self.get_transform_matrix()

        rgb = spec @ torch.from_numpy(M).float()
        return rgb
from tqdm import tqdm


def total_variation_in_third_dimension(img):
    return torch.mean((img[:,  1:] - img[:, :-1]) ** 2)

from math import ceil
import torch.nn as nn
import numpy as np
from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

def spectral_angle_mapper(vect_a, vect_b):
    """
    Calcula el Spectral Angle Mapper (SAM) entre dos vectores espectrales.
    
    Args:
        vect_a (torch.Tensor): El primer vector espectral (1D).
        vect_b (torch.Tensor): El segundo vector espectral (1D).
    
    Returns:
        sam (torch.Tensor): El ángulo espectral entre los dos vectores (en radianes).
    """
    # Normalizar los vectores
    norm_a = F.normalize(vect_a, p=2, dim=0)
    norm_b = F.normalize(vect_b, p=2, dim=0)
    
    # Calcular el producto punto entre los dos vectores normalizados
    dot_product = torch.sum(norm_a * norm_b)
    
    # Calcular el ángulo espectral (SAM)
    sam = torch.acos(torch.clamp(dot_product, min=-1.0, max=1.0))
    
    return sam

from math import ceil
import numpy as np
import torch
import torch.nn as nn

from math import ceil
import numpy as np
import torch
import torch.nn as nn

def get_manipulated_and_signed_signatures_subclasses(
    img: torch.Tensor,
    classes: np.ndarray,
    cs,
    idxs=None,
    max_signatures=1000,
    optimize_steps=900,
    lr=0.01,
    lambda_mse=1.0,
    lambda_sam=1.0
):
    H, W, C = img.shape
    device = img.device

    # 1) Filtrar sólo clases con píxeles
    unique, counts = np.unique(classes, return_counts=True)
    valid = unique[counts > 0]
    if valid.size < 2:
        raise RuntimeError("Se necesitan ≥2 clases con píxeles.")

    # 2) Elegir source/target válidos
    if idxs is None:
        src_cls, tgt_cls = np.random.choice(valid, size=2, replace=False)
    else:
        src_cls, tgt_cls = idxs
        if src_cls not in valid or tgt_cls not in valid:
            src_cls, tgt_cls = np.random.choice(valid, size=2, replace=False)

    # 3) Extraer firmas fuente
    src_idx = np.where(classes == src_cls)
    n_src = min(src_idx[0].shape[0], max_signatures)
    perm_src = np.random.permutation(src_idx[0].shape[0])[:n_src]
    signatures_source = torch.stack([
        img[src_idx[0][i], src_idx[1][i], :]
        for i in perm_src
    ], dim=0).to(device)  # [n_src, C]

    # 4) Extraer firmas objetivo
    tgt_idx = np.where(classes == tgt_cls)
    n_tgt = tgt_idx[0].shape[0]
    signatures_target = torch.stack([
        img[tgt_idx[0][i], tgt_idx[1][i], :]
        for i in range(n_tgt)
    ], dim=0).to(device)  # [n_tgt, C]

    # 5) Replicar y luego TRUNCAR a n_src
    if n_src > n_tgt and n_tgt > 0:
        reps = ceil(n_src / n_tgt)
        signatures_target = signatures_target.repeat(reps, 1)
    if signatures_target.shape[0] < n_src:
        raise RuntimeError("No hay suficientes firmas destino incluso tras replicar.")
    signatures_target = signatures_target[:n_src]  # ahora [n_src, C]

    # 6) Optimizar en dominio espectral
    x = nn.Parameter(signatures_target.clone(), requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)

    for _ in range(optimize_steps):
        optimizer.zero_grad()
        loss_mse = torch.mean((signatures_source - x)**2)
        sam = sum(spectral_angle_mapper(signatures_source[i], x[i]) for i in range(n_src))
        loss_sam = sam / n_src
        loss = lambda_mse * loss_mse + lambda_sam * loss_sam
        loss.backward()
        optimizer.step()

    # 7) Devolver sólo las firmas manipuladas y los índices fuente
    return x, signatures_source, (src_idx[0][perm_src], src_idx[1][perm_src])



from math import ceil
import torch.nn as nn
def get_manipulated_and_signed_signatures(img,classes,cs,idxs=None, max_signatures=1000,num_classes=6):
    if idxs is None:
        idxs = np.random.permutation(num_classes)[:2]
        source_class = idxs[0]
        target_class = idxs[1]
    else:
        source_class = idxs[0]
        target_class = idxs[1]

    indeces_source = np.where(classes==source_class)
    if indeces_source[0].shape[0] > max_signatures:
        indeces_source = (indeces_source[0][:max_signatures], indeces_source[1][:max_signatures])

    signatures_source = torch.zeros((indeces_source[0].shape[0], img.shape[2]))


    # storage the image values of class 5
    
    for i in range(indeces_source[0].shape[0]):

        signatures_source[i] = img[indeces_source[0][i], indeces_source[1][i], :]


    indeces_target = np.where(classes==target_class)
    signature_target = torch.zeros((indeces_target[0].shape[0], img.shape[2]))
    for i in range(indeces_target[0].shape[0]):
        signature_target[i] = img[indeces_target[0][i], indeces_target[1][i], :]
    

    if signatures_source.shape[0] > signature_target.shape[0]:
        num_signatures_to_replicate = ceil(signatures_source.shape[0] / signature_target.shape[0])
        signature_target = signature_target.repeat(num_signatures_to_replicate, 1)



    rgb_target = cs.spec_to_rgb(signature_target)

    rgb_source = cs.spec_to_rgb(signatures_source)

    x = nn.Parameter(torch.randn_like(signatures_source)*0.1+signatures_source.clone() , requires_grad=True)
    
    # plt.imshow(cs.spec_to_rgb(x).detach())
    # plt.title('Before optimization')
    # plt.show()
    optimizer = torch.optim.Adam([x], lr=0.01) 

    lambda_rgb = 1.0
    lambda_sam = 1.0
    # loop = tqdm(range(900), desc='Manipulating signatures',colour='green')
    for i in range(900):
        optimizer.zero_grad()
        rgb_manipulated = cs.spec_to_rgb(x)
        sam_v = lossimage_2(signature_target[:x.shape[0]], x)
        loss_rgb = torch.mean((rgb_source - rgb_manipulated) ** 2)
        loss_sam = sam_v**2
        tv = total_variation_in_third_dimension(x)
        loss = loss_rgb*lambda_rgb + loss_sam*lambda_sam + tv*0.1
        loss.backward()
        optimizer.step()
    
    return x, signatures_source,indeces_source

# wandb import removed
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_results_per_epoch(reconstructed, psnr, signed_img, manipulated_spectral_image, error, key, model_detecion, model_recovery, epoch, path, cs, img, H, W, loss_detection, loss_sensor, loss_recovery, indeces_source, manipulated, signed):
    psnr_v = psnr(reconstructed, signed_img.permute(2, 0, 1).unsqueeze(0))

    reconstructed_rgb = cs.spec_to_rgb(reconstructed.permute(0, 2, 3, 1).squeeze().detach().cpu())
    #     wandb.log({'loss_detection': loss_detection.item(), 'loss_sensor': loss_sensor.item(), 'loss_recovery': loss_recovery.item(), 'psnr': psnr_v.item()})

    rgb_manipulated = cs.spec_to_rgb(manipulated_spectral_image.permute(0, 2, 3, 1).squeeze().detach().cpu())
    #     wandb.log({'manipulated': [wandb.Image(rgb_manipulated.detach().cpu().numpy())]})
    #     wandb.log({'error': [wandb.Image((error / error.max()).detach().cpu().numpy())]})

    manipulated_pixels = torch.zeros((H, W))
    manipulated_pixels[indeces_source] = 1
    #     wandb.log({'Manipulated pixels': [wandb.Image((manipulated_pixels.numpy()))]})
    #     wandb.log({'reconstructed': [wandb.Image(reconstructed_rgb.detach().cpu().numpy())]})

    error = error.squeeze(0).squeeze(0)
    error = error / error.max()
    errorb = error > 0.1
    errorb = errorb.float()
    manipulated_pixels = manipulated_pixels.float().to('cuda')
    accuracy = torch.mean((errorb == manipulated_pixels).float())

    signatures_source = torch.zeros((indeces_source[0].shape[0], img.shape[2]))
    for i in range(indeces_source[0].shape[0]):
        signatures_source[i] = signed_img[indeces_source[0][i], indeces_source[1][i], :]

    # 1. Imagen de las firmas manipuladas (Manteniendo como estaba antes)
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(signed[:1000].squeeze().detach().cpu().numpy().T, linewidth=3)
    plt.title('Signed')
    plt.subplot(1, 3, 2)
    plt.plot(manipulated.squeeze().detach().cpu().numpy().T, linewidth=3)
    plt.title('Manipulated')
    plt.subplot(1, 3, 3)
    plt.plot(signatures_source.detach().cpu().numpy().T, linewidth=3)
    plt.savefig(f'{path}/manipulated_{epoch}.png')
    plt.close()

    # 2. Imagen de la matriz de llaves
    #plt.imshow(key.squeeze().detach().cpu().numpy(), cmap='cividis')
    #plt.colorbar()
    #plt.title('Key Matrix')
    #plt.savefig(f'{path}/key_matrix_{epoch}.png')
    #plt.close()

    # 3. Imagen de la media espacial de la llave
    key_mean = key.mean(dim=(0)).detach().cpu().numpy()
    plt.plot(key.squeeze().detach().cpu().numpy(), linewidth=3)
    plt.title('Spatial Mean of Key')
    plt.savefig(f'{path}/key_mean_{epoch}.png')
    plt.close()

    torch.save(key, f'{path}/key.pt')
    torch.save(model_detecion.state_dict(), f'{path}/model_detection.pt')
    torch.save(model_recovery.state_dict(), f'{path}/model_recovery.pt')

    #     # manual saving of images handled above; no wandb logging


# wandb import removed
import matplotlib.pyplot as plt
import torch
import numpy as np
def plot_results_per_epoch_classes(reconstructed, psnr, signed_img, manipulated_spectral_image, error, key, model_detection, model_recovery, epoch, path, cs, img, H, W, loss_detection, loss_sensor, loss_recovery, indeces_source, manipulated, signed, original_classes, subdivided_classes):
    psnr_v = psnr(reconstructed, signed_img.permute(2, 0, 1).unsqueeze(0))

    reconstructed_rgb = cs.spec_to_rgb(reconstructed.permute(0, 2, 3, 1).squeeze().detach().cpu())
    #     wandb.log({'loss_detection': loss_detection.item(), 'loss_sensor': loss_sensor.item(), 'loss_recovery': loss_recovery.item(), 'psnr': psnr_v.item()})

    rgb_manipulated = cs.spec_to_rgb(manipulated_spectral_image.permute(0, 2, 3, 1).squeeze().detach().cpu())
    #     wandb.log({'manipulated': [wandb.Image(rgb_manipulated.detach().cpu().numpy())]})
    #     wandb.log({'error': [wandb.Image((error / error.max()).detach().cpu().numpy())]})

    manipulated_pixels = torch.zeros((H, W))
    manipulated_pixels[indeces_source] = 1
    #     wandb.log({'Manipulated pixels': [wandb.Image((manipulated_pixels.numpy()))]})
    #     wandb.log({'reconstructed': [wandb.Image(reconstructed_rgb.detach().cpu().numpy())]})

    error = error.squeeze(0).squeeze(0)
    error = error / error.max()
    errorb = error > 0.1
    errorb = errorb.float()
    manipulated_pixels = manipulated_pixels.float().to('cuda')
    accuracy = torch.mean((errorb == manipulated_pixels).float())

    signatures_source = torch.zeros((indeces_source[0].shape[0], img.shape[2]))
    for i in range(indeces_source[0].shape[0]):
        signatures_source[i] = signed_img[indeces_source[0][i], indeces_source[1][i], :]

    # 1. Imagen de las firmas manipuladas (Manteniendo como estaba antes)
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(signed[:1000].squeeze().detach().cpu().numpy().T, linewidth=3)
    plt.title('Signed')
    plt.subplot(1, 3, 2)
    plt.plot(manipulated.squeeze().detach().cpu().numpy().T, linewidth=3)
    plt.title('Manipulated')
    plt.subplot(1, 3, 3)
    plt.plot(signatures_source.detach().cpu().numpy().T, linewidth=3)
    plt.savefig(f'{path}/manipulated_{epoch}.png')
    plt.close()

    # 2. Imagen de la matriz de llaves
    plt.imshow(key.squeeze().detach().cpu().numpy(), cmap='cividis')
    plt.colorbar()
    plt.title('Key Matrix')
    plt.savefig(f'{path}/key_matrix_{epoch}.png')
    plt.close()

    # 3. Imagen de la media espacial de la llave
    key_mean = key.mean(dim=(0)).detach().cpu().numpy()
    plt.plot(key_mean, linewidth=3)
    plt.title('Spatial Mean of Key')
    plt.savefig(f'{path}/key_mean_{epoch}.png')
    plt.close()

    # Guardar el mapa de clases original y subdividido
    plt.imshow(original_classes, cmap='tab10')
    plt.title('Original Classes Map')
    plt.savefig(f'{path}/original_classes_{epoch}.png')
    plt.close()

    plt.imshow(subdivided_classes, cmap='tab10')
    plt.title('Subdivided Classes Map')
    plt.savefig(f'{path}/subdivided_classes_{epoch}.png')
    plt.close()

    torch.save(key, f'{path}/key.pt')
    torch.save(model_detection.state_dict(), f'{path}/model_detection.pt')
    torch.save(model_recovery.state_dict(), f'{path}/model_recovery.pt')

    # manual logging removed




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

