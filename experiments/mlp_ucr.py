import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F

import warnings
warnings.filterwarnings('ignore')

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from aeon.datasets import load_classification

import numpy as np
import pandas as pd


from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from utils import load_data, to_torch_dataset, to_torch_loader


class Sine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return torch.sin(x)


class Snake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return torch.sin(x) ** 2 + x


class LeakySineLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return torch.where(x > 0, torch.sin(x) ** 2 + x, 0.5 * torch.sin(x) ** 2 + x)


class MLP(LightningModule):
    def __init__(self, sequence_length: int, in_channels: int, num_classes: int, activation: nn.Module, activation_kw: dict = {}) -> None:
        super().__init__()
        self.sequence_len = sequence_length
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.activation = activation
        
        self.layers = nn.Sequential(*[
            nn.Flatten(),
            nn.Dropout1d(p=0.1),
            nn.Linear(in_features=sequence_length * in_channels, out_features=500),
            activation(**activation_kw),
            
            nn.Dropout1d(p=0.2),
            nn.Linear(in_features=500, out_features=500),
            activation(**activation_kw),
            
            nn.Dropout1d(p=0.2),
            nn.Linear(in_features=500, out_features=500),
            activation(**activation_kw),
            
            nn.Dropout1d(p=0.3),
            nn.Linear(in_features=500, out_features=1 if num_classes == 2 else num_classes),
        ])

        self.criteria = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adadelta(self.parameters(), lr=1e-1, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=200, min_lr=0.1
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = F.sigmoid(preds).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

        loss = self.criteria(preds, y.float().to(self.device) if self.num_classes == 2 else y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)            

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_f1', f1, on_epoch=True, on_step=False)
        self.log('train_precision', precision, on_epoch=True, on_step=False)
        self.log('train_recall', recall, on_epoch=True, on_step=False)
    
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)

        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = F.sigmoid(preds).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        self.log('acc', acc)
        self.log('f1', f1)
        self.log('precision', precision)
        self.log('recall', recall)

        return
    
DATASETS = [
    # 'Adiac',
    # 'ArrowHead',
    # 'Beef',
    # 'BeetleFly',
    # 'BirdChicken',
    # 'Car',
    # 'CBF',
    # 'ChlorineConcentration',
    # 'CinCECGTorso',
    # 'Coffee',
    # 'Computers',
    # 'CricketX',
    # 'CricketY',
    # 'CricketZ',
    # 'DiatomSizeReduction',
    # 'DistalPhalanxOutlineAgeGroup',
    # 'DistalPhalanxOutlineCorrect',
    # 'DistalPhalanxTW',
    # 'Earthquakes',
    # 'ECG200',
    # 'ECG5000',
    # 'ECGFiveDays',
    # 'ElectricDevices',
    # 'FaceAll',
    # 'FaceFour',
    # 'FacesUCR',
    # 'FiftyWords',
    # 'Fish',
    # 'FordA',
    # 'FordB',
    # 'GunPoint',
    # 'Ham',
    # 'HandOutlines',
    # 'Haptics',
    # 'Herring',
    # 'InlineSkate',
    # 'InsectWingbeatSound',
    # 'ItalyPowerDemand',
    # 'LargeKitchenAppliances',
    # 'Lightning2',
    # 'Lightning7',
    # 'Mallat',
    # 'Meat',
    # 'MedicalImages',
    # 'MiddlePhalanxOutlineAgeGroup',
    # 'MiddlePhalanxOutlineCorrect',
    # 'MiddlePhalanxTW',
    # 'MoteStrain',
    # 'NonInvasiveFetalECGThorax1',
    # 'NonInvasiveFetalECGThorax2',
    # 'OliveOil',
    # 'OSULeaf',
    # 'PhalangesOutlinesCorrect',
    # 'Phoneme',
    # 'Plane',
    # 'ProximalPhalanxOutlineAgeGroup',
    # 'ProximalPhalanxOutlineCorrect',
    # 'ProximalPhalanxTW',
    # 'RefrigerationDevices',
    # 'ScreenType',
    # 'ShapeletSim',
    # 'ShapesAll',
    # 'SmallKitchenAppliances',
    # 'SonyAIBORobotSurface1',
    # 'SonyAIBORobotSurface2',
    # 'StarLightCurves',
    # 'Strawberry',
    # 'SwedishLeaf',
    # 'Symbols',
    # 'SyntheticControl',
    # 'ToeSegmentation1',
    # 'ToeSegmentation2',
    # 'Trace',
    # 'TwoLeadECG',
    # 'TwoPatterns',
    # 'UWaveGestureLibraryAll',
    # 'UWaveGestureLibraryX',
    # 'UWaveGestureLibraryY',
    # 'UWaveGestureLibraryZ',
    # 'Wafer',
    # 'Wine',
    # 'WordSynonyms',
    # 'Worms',
    # 'WormsTwoClass',
    # 'Yoga',
    # 'ACSF1',
    # 'AllGestureWiimoteX',
    # 'AllGestureWiimoteY',
    # 'AllGestureWiimoteZ',
    # 'BME',
    # 'Chinatown',
    # 'Crop',
    # 'DodgerLoopDay',
    # 'DodgerLoopGame',
    # 'DodgerLoopWeekend',
    # 'EOGHorizontalSignal',
    # 'EOGVerticalSignal',
    # 'EthanolLevel',
    # 'FreezerRegularTrain',
    # 'FreezerSmallTrain',
    # 'Fungi',
    # 'GestureMidAirD1',
    # 'GestureMidAirD2',
    # 'GestureMidAirD3',
    # 'GesturePebbleZ1',
    # 'GesturePebbleZ2',
    # 'GunPointAgeSpan',
    # 'GunPointMaleVersusFemale',
    # 'GunPointOldVersusYoung',
    # 'HouseTwenty',
    # 'InsectEPGRegularTrain',
    # 'InsectEPGSmallTrain',
    # 'MelbournePedestrian',
    # 'MixedShapesRegularTrain',
    # 'MixedShapesSmallTrain',
    # 'PickupGestureWiimoteZ',
    # 'PigAirwayPressure',
    # 'PigArtPressure',
    # 'PigCVP',
    # 'PLAID',
    'PowerCons',
    'Rock',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'SmoothSubspace',
    'UMD'
]

NUMBER_OF_EXPERIMENTS = 1
NUMBER_OF_EPOCHS = 1000

results_data_dir = {
    'model': [],
    'dataset': [],
    'activation': [],
    'exp': [],
    'acc': [],
    'f1': []
}

activations_dict = {
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'sine': Sine,
    'leakysinelu': LeakySineLU,
    'snake': Snake,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'elu': nn.ELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid
}

for dataset in DATASETS:
    for activation in ['relu', 'prelu', 'sine', 'leakysinelu', 'snake', 'gelu', 'silu', 'elu', 'tanh', 'sigmoid']:

        print(f'Loading dataset {dataset}...')
        X_train, y_train, X_test, y_test = load_data(name=dataset, task='classification', split='full')
        
        print('Converting the dataset to torch.DataLoader...')
        train_set, test_set = to_torch_dataset(X_train, y_train, X_test, y_test)
        train_loader, test_loader = to_torch_loader(train_dataset=train_set, test_dataset=test_set)

        num_classes = len(np.unique(y_train))

        for experiment_number in range(NUMBER_OF_EXPERIMENTS):
            model = MLP(sequence_length=X_train.shape[-1],
                        in_channels=1,
                        num_classes=num_classes,
                        activation=activations_dict[activation])

            checkpoint = ModelCheckpoint(
                monitor='train_loss',
                dirpath=f'./models',
                filename=f'mlp-{dataset}-{experiment_number}-{activation}',
                save_top_k=1,
                auto_insert_metric_name=False
            )

            trainer = Trainer(
                max_epochs=NUMBER_OF_EPOCHS,
                accelerator='gpu',
                devices=-1,
                callbacks=[checkpoint]
            )
            
            trainer.fit(model, train_dataloaders=train_loader)
            results = trainer.test(model, test_loader)
            
            
            
            results_data_dir['dataset'].append(dataset)
            results_data_dir['model'].append('mlp')
            results_data_dir['exp'].append(experiment_number)
            results_data_dir['acc'].append(results[0]['acc'])
            results_data_dir['f1'].append(results[0]['f1'])
            results_data_dir['activation'].append(activation)

            results_df = pd.DataFrame(results_data_dir)
            results_df.to_csv(f'./mlp.csv', index=False)
