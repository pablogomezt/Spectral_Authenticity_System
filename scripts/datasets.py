import os
import ast
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import TensorDataset


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)

    # Central region

    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape

    # Mirror left

    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]

    # Mirror on the right

    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[
                                                                                                        :, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[
                                                                                                        :, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    return x_train_band


def choose_subset_point(train_data, val_data, num_classes):
    number_train = []
    pos_train = {}
    number_val = []
    pos_val = {}

    # for train data

    for i in range(num_classes):
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)

    # for val data

    for i in range(num_classes):
        each_class = np.argwhere(val_data == (i + 1))
        number_val.append(each_class.shape[0])
        pos_val[i] = each_class

    total_pos_val = pos_val[0]
    for i in range(1, num_classes):
        total_pos_val = np.r_[total_pos_val, pos_val[i]]

    return total_pos_train, total_pos_val, number_train, number_val


def subset_data(mirror_image, band, train_point, val_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_val = np.zeros((val_point.shape[0], patch, patch, band), dtype=float)

    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(val_point.shape[0]):
        x_val[j, :, :, :] = gain_neighborhood_pixel(mirror_image, val_point, j, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_val  shape = {}, type = {}".format(x_val.shape, x_val.dtype))
    print("**************************************************")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_val_band = gain_neighborhood_band(x_val, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape, x_train_band.dtype))
    print("x_val_band  shape = {}, type = {}".format(x_val_band.shape, x_val_band.dtype))
    print("**************************************************")
    return x_train_band, x_val_band


def subset_label(number_train, number_val, num_classes):
    y_train = []
    y_val = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for v in range(number_val[i]):
            y_val.append(i)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    print("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))
    print("y_val: shape = {} ,type = {}".format(y_val.shape, y_val.dtype))
    print("**************************************************")
    return y_train, y_val


def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)

    # Central region

    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize

    # Mirror left

    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]

    # Mirror on the right

    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]

    # Mirror above

    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]

    # Mirror below

    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1],
                                                      mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


def split_dataset(labels, split, seed):
    np.random.seed(seed)
    train_labels = np.zeros(labels.shape)
    val_labels = np.zeros(labels.shape)

    for i in range(1, int(np.max(labels)) + 1):
        class_indices = np.argwhere(labels == i)
        np.random.shuffle(class_indices)
        split_index = int(split['train'] * len(class_indices))
        train_indices = class_indices[:split_index]
        val_indices = class_indices[split_index:]
        train_labels[train_indices[:, 0], train_indices[:, 1]] = i
        val_labels[val_indices[:, 0], val_indices[:, 1]] = i

    return train_labels, val_labels


def load_preprocessed_dataset(data_config):
    if data_config['name'].lower() == 'indian_pines':
        hsi = sio.loadmat(os.path.join(data_config['path'], 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        label = sio.loadmat(os.path.join(data_config['path'], 'Indian_pines_gt.mat'))['indian_pines_gt']
        data = dict(hsi=hsi, label=label)
    elif data_config['name'].lower() == 'pavia_university':
        data = sio.loadmat(os.path.join(data_config['path'], 'Pavia.mat'))
    elif data_config['name'].lower() == 'houston':
        data = sio.loadmat(os.path.join(data_config['path'], 'Houston.mat'))
    else:
        raise ValueError(f"Unknown dataset: {data_config['name']}")

    TR, VA = split_dataset(data['label'], data_config['split'], data_config['seed'])

    hsi = data['hsi']
    num_classes = int(np.max(TR))

    hsi_normalize = np.zeros(hsi.shape)
    for i in range(hsi.shape[2]):
        hsi_max = np.max(hsi[:, :, i])
        hsi_min = np.min(hsi[:, :, i])
        if hsi_max != hsi_min:
            hsi_normalize[:, :, i] = (hsi[:, :, i] - hsi_min) / (hsi_max - hsi_min)
    # data size
    height, width, band = hsi.shape
    print("height={0},width={1},band={2}".format(height, width, band))

    # obtain train and test data

    total_pos_train, total_pos_val, number_train, number_val = choose_subset_point(TR, VA, num_classes)
    mirror_image = mirror_hsi(height, width, band, hsi_normalize, patch=data_config['patches'])
    x_train_band, x_val_band = subset_data(mirror_image, band, total_pos_train, total_pos_val,
                                           patch=data_config['patches'], band_patch=data_config['band_patches'])
    y_train, y_val = subset_label(number_train, number_val, num_classes)

    # load data

    x_train = torch.from_numpy(x_train_band.transpose(0, 2, 1)).float()
    y_train = torch.from_numpy(y_train).long()
    train_dataset = TensorDataset(x_train, y_train)

    x_val = torch.from_numpy(x_val_band.transpose(0, 2, 1)).float()
    y_val = torch.from_numpy(y_val).long()
    val_dataset = TensorDataset(x_val, y_val)

    return train_dataset, val_dataset, (band, num_classes)
