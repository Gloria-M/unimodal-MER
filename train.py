import os
import numpy as np
import torch
import torch.nn as nn

from models import AudioNet
from data_loader import make_training_loaders
from utility_functions import *


class Trainer:
    def __init__(self, args):

        self.dimension = args.dimension

        self._data_dir = args.data_dir
        self._models_dir = args.models_dir
        self._plots_dir = args.plots_dir
        self._device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        self._lr = args.lr_init
        self._lr_decay = args.lr_decay
        self._weight_decay = args.weight_decay

        self.train_loader, self.test_loader = make_training_loaders(self._data_dir)

        if self.dimension == 'valence':
            self._params_dict = args.valence_params_dict
        elif self.dimension == 'arousal':
            self._params_dict = args.arousal_params_dict
        else:
            self._params_dict = args.params_dict

        self.model = AudioNet(self._params_dict).to(self._device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        self._criterion = nn.MSELoss()

        if self.dimension == 'both':
            self.train_dict = {'valence_loss': [], 'arousal_loss': []}
            self.test_dict = {'valence_loss': [], 'arousal_loss': []}
        else:
            self.train_dict = {'loss': []}
            self.test_dict = {'loss': []}

    def save_model(self):

        model_path = os.path.join(self._models_dir, 'model_{:s}.pt'.format(self.dimension))
        torch.save(self.model.state_dict(), model_path)

    def update_learning_rate(self):

        self._lr *= self._lr_decay

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._lr

        success_message = 'Learning rate updated to {:.1e}'.format(self._lr)
        print(success_format(success_message))

    def train_1d(self):

        train_loss = []

        self.model.train()
        for batch_idx, (data, annotations) in enumerate(self.train_loader):

            if self.dimension == 'valence':
                target = annotations[:, 0]
            elif self.dimension == 'arousal':
                target = annotations[:, 1]

            data = data.to(self._device)
            target = target.to(self._device)

            self.optimizer.zero_grad()
            output = self.model(data)

            target = target.view_as(output)

            batch_loss = self._criterion(output, target)
            batch_loss.backward()
            train_loss.append(batch_loss.data.cpu().numpy())

            self.optimizer.step()

        self.train_dict['loss'].append(np.array(train_loss).mean())

    def train_2d(self):

        true_annotations = []
        pred_annotations = []

        self.model.train()
        for batch_idx, (data, annotations) in enumerate(self.train_loader):

            data = data.to(self._device)
            annotations = annotations.to(self._device)

            self.optimizer.zero_grad()
            output = self.model(data)

            true_annotations.extend(annotations.cpu().detach().numpy())
            pred_annotations.extend(output.cpu().detach().numpy())

            batch_loss = self._criterion(output, annotations)
            batch_loss.backward()

            self.optimizer.step()

        true_annotations = np.array(true_annotations)
        pred_annotations = np.array(pred_annotations)

        true_valence = np.array([annot[0] for annot in true_annotations])
        pred_valence = np.array([annot[0] for annot in pred_annotations])
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        self.train_dict['valence_loss'].append(valence_mse)
        self.train_dict['arousal_loss'].append(arousal_mse)

    def validate_1d(self):

        test_loss = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, annotations) in enumerate(self.test_loader):

                if self.dimension == 'valence':
                    target = annotations[:, 0]
                elif self.dimension == 'arousal':
                    target = annotations[:, 1]

                data = data.to(self._device)
                target = target.to(self._device)

                output = self.model(data)

                target = target.view_as(output)

                batch_loss = self._criterion(output, target)
                test_loss.append(batch_loss.data.cpu().numpy())

        self.test_dict['loss'].append(np.array(test_loss).mean())

    def validate_2d(self):

        true_annotations = []
        pred_annotations = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, annotations) in enumerate(self.test_loader):

                data = data.to(self._device)
                annotations = annotations.to(self._device)

                output = self.model(data)

                true_annotations.extend(annotations.cpu().detach().numpy())
                pred_annotations.extend(output.cpu().detach().numpy())

        true_annotations = np.array(true_annotations)
        pred_annotations = np.array(pred_annotations)

        true_valence = np.array([annot[0] for annot in true_annotations])
        pred_valence = np.array([annot[0] for annot in pred_annotations])
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        self.test_dict['valence_loss'].append(valence_mse)
        self.test_dict['arousal_loss'].append(arousal_mse)
