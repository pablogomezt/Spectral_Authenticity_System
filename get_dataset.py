import torch
import scipy.io as sio
import numpy as np
import os


def get_dataset(args):
    if args.dataset == 'indian_pines':
        img = sio.loadmat('datasets/Indian_pines.mat')['indian_pines_corrected']
        classes = sio.loadmat('datasets/Indian_pines_gt.mat')['indian_pines_gt']
    elif args.dataset == 'indian_pines_crop':
        img = sio.loadmat('datasets/cropped_cube.mat')['cropped_cube']
        classes = sio.loadmat('datasets/cropped_class_image.mat')['cropped_class_image']
    elif args.dataset == 'multi':
        img = sio.loadmat('datasets/pavia_multispectral.mat')['multispectral_cube']
        classes = sio.loadmat('datasets/Pavia_gt.mat')['pavia_gt']
    elif args.dataset == 'pavia':
        img = sio.loadmat('datasets/Pavia.mat')['pavia']
        classes = sio.loadmat('datasets/Pavia_gt.mat')['pavia_gt']
    elif args.dataset == 'salinas':
        img = sio.loadmat('datasets/Salinas.mat')['salinas_corrected']
        classes = sio.loadmat('datasets/Salinas_gt.mat')['salinas_gt']
    elif args.dataset == 'circular':
        img = sio.loadmat('datasets/circular.mat')['circular']
        classes = sio.loadmat('datasets/circular_gt.mat')['circular_gt']-1
    elif args.dataset == 'cuadrada':
        img = sio.loadmat('datasets/cuadrada.mat')['cuadrada']
        classes = sio.loadmat('datasets/cuadrada_gt.mat')['cuadrada_gt']

    img = img.astype(np.float32)
    img = img / np.max(img)
    img = torch.from_numpy(img)
    return img, classes

