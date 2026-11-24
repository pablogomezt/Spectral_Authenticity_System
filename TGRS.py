import torch
import scipy.io as sio
import numpy as np
import os
from scripts.utils import ColourSystem

from scripts.utils import get_manipulated_and_signed_signatures

import matplotlib.pyplot as plt
from scripts.utils import plot_results_per_epoch

from math import ceil
import argparse
import torch.nn as nn
from torchmetrics.image import SpectralAngleMapper
from torchmetrics import PeakSignalNoiseRatio

from tqdm import tqdm

from models_2 import Autoencoder, UNet

from losses import *
from get_dataset import get_dataset

WAVELENGTHS = {'pavia': [430,860],
               'indian_pines': [400, 2500],
               'salinas': [400, 2500],
               'circular': [400, 700],
               'cuadrada': [400, 700],
               'cuadrada_mixedv2': [400, 700],
               'color_mixed': [400, 700]
               }
def main(args):
    
    img, classes = get_dataset(args)
    img = img.to('cuda')
    H, W, C = img.shape

    num_classes = len(np.unique(classes))
    print(num_classes)
    
    path = f'results/TGRS_{args.dataset}_epochs_{args.epochs}_beta_{args.beta}_lambda_{args.lambda_}_eta_{args.eta}'

    w_start = WAVELENGTHS[args.dataset][0]
    w_end = WAVELENGTHS[args.dataset][1]
    
    cs = ColourSystem(start=w_start, end=w_end, num=img.shape[2])
    if not os.path.exists(path):
        os.makedirs(path)   
    # Inicialización aleatoria de la key
    key = torch.nn.Parameter(torch.rand((1, 1, img.shape[2])).to('cuda') * 0.2, requires_grad=True)

    model_detecion = Autoencoder(input_size=img.shape[2], hidden_size=100, latent_size=10).to('cuda')
    model_recovery = UNet(in_channels=img.shape[2], out_channels=img.shape[2]).to('cuda')   

    # include key in optimizer so it gets updated during training
    optimizer_t = torch.optim.Adam(list(model_detecion.parameters()) + list(model_recovery.parameters()) + [key], lr=0.005)

    # wandb initialization removed
    print(args.epochs)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda')

    for epoch in range(args.epochs):
        signed_img = img + key
        signed_img = torch.clamp(signed_img, 0, 0.99)
        signed_img = signed_img.to('cuda')

        idxs = np.random.permutation(num_classes)[:2]
        source_class = idxs[0]
        target_class = idxs[1]
        print(source_class, target_class)
        manipulated, signed, indeces_source = get_manipulated_and_signed_signatures(img, classes, cs, idxs=[source_class, target_class], max_signatures=args.max_signatures)
        manipulated = manipulated.to('cuda')
        signed = signed.to('cuda')

        loop = tqdm(range(args.number_steps), total=args.number_steps, colour='green', desc=f'Epoch {epoch}')
        
        for i in loop:
            optimizer_t.zero_grad()
            
            key_estimated_manipulated = model_detecion(manipulated)
            key_estimated_signed = model_detecion(signed)

            manipulated_spectral_image = img.clone()
            manipulated_spectral_image[indeces_source] = manipulated
            maniputlated_spectral_image_vect = manipulated_spectral_image.reshape(-1, img.shape[2])

            key_manipulated = model_detecion(maniputlated_spectral_image_vect)

            error = ((key_manipulated - key.squeeze(0))**2).mean(dim=-1)
            error = error.reshape(H, W)
            error = error.unsqueeze(0).unsqueeze(0)

            manipulated_spectral_image = manipulated_spectral_image.permute(2, 0, 1).unsqueeze(0)
            reconstructed = model_recovery(manipulated_spectral_image, error)
            reconstructed = torch.clamp(reconstructed, 0, 1)

            loss_signed = torch.norm(key_estimated_signed - key.squeeze(0).repeat(key_estimated_signed.shape[0], 1), dim=1).mean() / torch.norm(key)
            loss_manipulated = (1 - torch.norm(key_estimated_manipulated - key.squeeze(0).repeat(key_estimated_manipulated.shape[0], 1), dim=1).mean() / torch.norm(key))**2
            loss_detection = loss_signed + loss_manipulated

            loss_recovery = torch.mean((reconstructed - signed_img.permute(2, 0, 1).unsqueeze(0))**2)
            loss_sensor = torch.mean((img - signed_img)**2)

            losst = args.beta * loss_detection + loss_recovery * args.lambda_ + loss_sensor * args.eta
            
            losst.backward(retain_graph=True)
            optimizer_t.step()

        plot_results_per_epoch(reconstructed, psnr, signed_img, manipulated_spectral_image, error, key, model_detecion, model_recovery, epoch, path, cs, img, H, W, loss_detection, loss_sensor, loss_recovery, indeces_source, manipulated, signed)

if __name__ == '__main__':
    import argparse
    from itertools import product

    parser = argparse.ArgumentParser(description='Training script for TGRS modulation and recovery')
    parser.add_argument('--target_class', default=5, type=float, help='target class')
    parser.add_argument('--source_class', default=10, type=float, help='source class')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--signed', default=True, type=bool, help='signed')
    parser.add_argument('--key_val', default='07', type=str, help='key_val')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--dataset', default='salinas', type=str, help='dataset')
    parser.add_argument('--max_signatures', default=1000, type=int)
    parser.add_argument('--model', default='autoencoder', type=str)
    parser.add_argument('--number_steps', default=1, type=int)
    parser.add_argument('--beta', default=10.0, type=float, help='beta for detection loss')
    parser.add_argument('--lambda_', default=10.0, type=float, help='lambda for recovery loss')
    parser.add_argument('--eta', default=0.1, type=float, help='eta for sensor loss')

    args = parser.parse_args()
    main(args)