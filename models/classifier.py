import os
from colorama import Fore

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchmetrics

import lightning as L

from scripts.metrics import output_metric


class Classifier(L.LightningModule):
    def __init__(self, num_classes,
                 max_epochs,
                 datasets,
                 classifier=None,
                 learning_rate=1e-3,
                 lr_params=None,
                 save_metrics_each=0,
                 save_dir=None):

        super().__init__()
        assert classifier is not None, 'Classifier is required'
        assert isinstance(datasets, dict), ('Datasets must be a dictionary with `train_dataset` and `val_dataset` keys')

        self.train_dataset = datasets['train_dataset']
        self.val_dataset = datasets['val_dataset']

        self.num_classes = num_classes
        self.classifier = classifier
        self.init_lr = learning_rate
        self.lr_params = lr_params
        self.max_epochs = max_epochs
        self.save_metrics_each = save_metrics_each
        self.save_dir = save_dir

        self.ce_loss = nn.CrossEntropyLoss()

        self.acc = torchmetrics.functional.accuracy
        self.prec = torchmetrics.functional.precision
        self.reca = torchmetrics.functional.recall
        self.f1 = torchmetrics.functional.f1_score

    def general_step(self, mode, batch, batch_idx):
        x, c = batch

        c_hat = self.classifier(x)
        c_hat = c_hat[None, ...] if c_hat.ndim == 1 else c_hat
        loss = self.ce_loss(c_hat, c)

        # regularizers

        loss_total = loss
        try:
            loss_reg = self.classifier.apply_l2_regularization()
        except:
            loss_reg = 0

        loss_total += loss_reg

        # metrics

        acc_loss = self.acc(c_hat, c, task='multiclass', num_classes=self.num_classes)
        prec_loss = self.prec(c_hat, c, task='multiclass', num_classes=self.num_classes)
        reca_loss = self.reca(c_hat, c, task='multiclass', num_classes=self.num_classes)
        f1_loss = self.f1(c_hat, c, task='multiclass', num_classes=self.num_classes)

        # logger

        self.log_dict({f'total_loss/{mode}': loss_total,
                       f'acc_loss/{mode}': acc_loss,
                       f'loss/{mode}': loss,
                       f'prec/{mode}': prec_loss,
                       f'reca/{mode}': reca_loss,
                       f'f1/{mode}': f1_loss,
                       }, on_step=False, on_epoch=True)

        # figure

        if mode == 'val':
            if batch_idx == 0 and self.current_epoch % self.save_metrics_each == 0:
                wavelengths = np.linspace(0, x.shape[1] - 1, x.shape[1]).astype(np.float32)

                if x.ndim == 3:
                    x = x[..., 0]
                elif x.ndim == 4:
                    x = x[..., 0, 0]

                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                markerline, stemlines, baseline = plt.stem(wavelengths, x[0].detach().cpu().numpy().squeeze(), 'o-')
                plt.setp(stemlines, linewidth=1)
                plt.setp(markerline, markersize=2)
                plt.title('Spectral Signature')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(1, 2, 2)
                plot_value_array(c_hat[0].detach().cpu().numpy(), c[0].detach().cpu().numpy())
                plt.title('Predictions')
                plt.xlabel('Class')
                plt.ylabel('Probability')

                plt.tight_layout()
                out_name = os.path.join(self.save_dir, self.logger.name, f'val_epoch{self.current_epoch + 1}')
                plt.savefig(out_name)
                plt.close()

        return loss_total

    def training_step(self, batch, batch_idx):
        if batch_idx == 160:
            print('')
        return self.general_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.general_step('val', batch, batch_idx)

    def metric_step(self, mode, dataset):
        self.classifier.eval()
        with torch.no_grad():
            c_list = []
            c_hat_list = []
            with torch.no_grad():
                for batch in dataset:
                    x, c = batch

                    x = x.to(self.device)
                    c = c.to(self.device)

                    c_hat = self.classifier(x)
                    c_hat = c_hat[None, ...] if c_hat.ndim == 1 else c_hat
                    c_hat = torch.argmax(c_hat, dim=1)

                    c_list.append(c)
                    c_hat_list.append(c_hat)

            c_list = torch.cat(c_list).detach()
            c_hat_list = torch.cat(c_hat_list).detach()

            oa_loss, aa_loss, kappa_loss, _ = output_metric(c_list, c_hat_list, num_classes=self.num_classes)

            self.log_dict({f'oa_loss/{mode}': oa_loss,
                           f'aa_loss/{mode}': aa_loss,
                           f'kappa_loss/{mode}': kappa_loss,
                           }, on_step=False, on_epoch=True)

            print(f"{Fore.YELLOW}{mode.capitalize()} -> "
                  f"{Fore.GREEN}oa_loss {oa_loss.item():.4f} - aa_loss {aa_loss.item():.4f} - "
                  f"kappa_loss {kappa_loss.item():.4f}")

    def ml_metric_step(self):
        # train and val dataset for ml metrics

        train_samples = self.train_dataset.dataset[:]
        x_train, c_train = train_samples[0], train_samples[1]

        val_samples = self.val_dataset.dataset[:]
        x_val, c_val = val_samples[0], val_samples[1]

        x_train = x_train.squeeze().cpu().numpy()
        c_train = c_train.cpu().numpy().astype(np.uint8)

        x_val = x_val.squeeze().cpu().numpy()
        c_val = c_val.cpu().numpy().astype(np.uint8)

        if x_train.ndim == 3:
            x_train = x_train[..., x_train.shape[-1] // 2]
            x_val = x_val[..., x_val.shape[-1] // 2]

        # svm and knn classification

        estimators = dict(knn=KNeighborsClassifier(n_neighbors=3),
                          svm=SVC(kernel='rbf', C=1e5, gamma=1.))

        for key, estimator in estimators.items():
            estimator.fit(x_train, c_train)
            c_hat_val = estimator.predict(x_val)

            oa_loss_val, aa_loss_val, kappa_loss_val, _ = output_metric(torch.from_numpy(c_val),
                                                                        torch.from_numpy(c_hat_val),
                                                                        num_classes=self.num_classes)

            self.log_dict({f'oa_loss/{key}_val': oa_loss_val,
                           f'aa_loss/{key}_val': aa_loss_val,
                           f'kappa_loss/{key}_val': kappa_loss_val,
                           }, on_step=False, on_epoch=True)

            print(f"{Fore.YELLOW}{key.upper()} - val  -> "
                  f"{Fore.GREEN}oa_loss {oa_loss_val.item():.4f} - aa_loss {aa_loss_val.item():.4f} - "
                  f"kappa_loss {kappa_loss_val.item():.4f}")

    def on_train_epoch_end(self):
        if int(self.current_epoch + 1) % self.save_metrics_each == 0:
            self.metric_step('train', self.train_dataset)
            self.metric_step('val ', self.val_dataset)

    def on_train_end(self):
        self.ml_metric_step()

    def configure_optimizers(self):
        weight_decay = 0 if 'weight_decay' not in list(self.lr_params['params'].keys()) \
            else self.lr_params['params']['weight_decay']
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.init_lr), weight_decay=weight_decay)

        if self.lr_params['scheduler'] is None:
            return dict(optimizer=optimizer)

        else:
            self.lr_params['params']['scheduler'].pop('name')
            scheduler = self.lr_params['scheduler'](optimizer, **self.lr_params['params']['scheduler'])
            lr_scheduler = dict(scheduler=scheduler)
            if 'monitor' in list(self.lr_params['params'].keys()):
                lr_scheduler['monitor'] = self.lr_params['params']['monitor']

            return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)


def plot_value_array(predictions_array, true_label):
    num_classes = len(predictions_array) + 1
    plt.grid(False)
    plt.xticks(range(1, num_classes))
    plt.yticks([])
    thisplot = plt.bar(range(1, num_classes), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
