import ast
import warnings

from utils.tools import str2bool

warnings.filterwarnings("ignore")
import argparse
import copy
import multiprocessing
from typing import Optional, Union
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

try:
    import xarray as xr
except ImportError:
    xr = None

d2r = np.pi / 180


def latlon2xyz(lat, lon):
    if type(lat) == torch.Tensor:
        x = -torch.cos(lat)*torch.cos(lon)
        y = -torch.cos(lat)*torch.sin(lon)
        z = torch.sin(lat)

    if type(lat) == np.ndarray:
        x = -np.cos(lat)*np.cos(lon)
        y = -np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
    return x, y, z


def xyz2latlon(x, y, z):
    if type(x) == torch.Tensor:
        lat = torch.arcsin(z)
        lon = torch.atan2(-y, -x)

    if type(x) == np.ndarray:
        lat = np.arcsin(z)
        lon = np.arctan2(-y, -x)
    return lat, lon


data_map = {
    'g': 'geopotential',
    'z': 'geopotential_500',
    't': 'temperature',
    't850': 'temperature_850',
    'tp': 'total_precipitation',
    't2m': '2m_temperature',
    'r': 'relative_humidity',
    'u10': '10m_u_component_of_wind',
    'u': 'u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'v': 'v_component_of_wind',
    'tcc': 'total_cloud_cover',
}

mv_data_map = {
    **dict.fromkeys(['mv', 'mv4'], ['r', 't', 'u', 'v']),
    'mv5': ['g', 'r', 't', 'u', 'v'],
}


class WeatherBenchDataset(Dataset):
    """Wheather Bench Dataset <http://arxiv.org/abs/2002.00469>`_

    Args:
        data_root (str): Path to the dataset.
        data_name (str): Name of the weather modality in Wheather Bench.
        training_time (list): The arrange of years for training.
        idx_in (list): The list of input indices.
        idx_out (list): The list of output indices to predict.
        step (int): Sampling step in the time dimension.
        level (int): Used level in the multi-variant version.
        data_split (str): The resolution (degree) of Wheather Bench splits.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, data_name, training_time,
                 idx_in, idx_out, step=1, level=1, data_split='5_625',
                 mean=None, std=None, use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.data_name = data_name
        self.data_split = data_split
        self.training_time = training_time
        self.idx_in = np.array(idx_in)
        self.idx_out = np.array(idx_out)
        self.step = step
        self.level = level
        self.data = None
        self.mean = mean
        self.std = std
        self.use_augment = use_augment
        assert isinstance(level, (int, list))

        self.time = None
        shape = int(32 * 5.625 / float(data_split.replace('_', '.')))
        self.shape = (shape, shape * 2)

        if isinstance(data_name, list):
            data_name = data_name[0]
        if 'mv' in data_name:  # multi-variant version
            self.data_name = mv_data_map[data_name]
            self.data, self.mean, self.std = [], [], []
            for name in self.data_name:
                data, mean, std = self._load_data_xarray(data_name=name, single_variant=False)
                self.data.append(data)
                self.mean.append(mean)
                self.std.append(std)
            self.data = np.concatenate(self.data, axis=1)
            self.mean = np.concatenate(self.mean, axis=1)
            self.std = np.concatenate(self.std, axis=1)
        else:  # single variant
            self.data_name = data_name
            self.data, mean, std = self._load_data_xarray(data_name, single_variant=True)
            if self.mean is None:
                self.mean, self.std = mean, std

        self.valid_idx = np.array(
            range(-idx_in[0], self.data.shape[0]-idx_out[-1]-1))

    def _load_data_xarray(self, data_name, single_variant=True):
        """Loading full data with xarray"""
        if data_name != 'uv10':
            try:
                dataset = xr.open_mfdataset(self.data_root+'/{}/{}*.nc'.format(
                    data_map[data_name], data_map[data_name]), combine='by_coords')
            except (AttributeError, ValueError):
                assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                    'pip install xarray==0.19.0,' \
                                    'pip install netcdf4 h5netcdf dask'
            except OSError:
                print("OSError: Invalid path {}/{}/*.nc".format(self.data_root, data_map[data_name]))
                assert False
            dataset = dataset.sel(time=slice(*self.training_time))
            dataset = dataset.isel(time=slice(None, -1, self.step))
            if self.time is None and single_variant:
                self.week = dataset['time.week']
                self.month = dataset['time.month']
                self.year = dataset['time.year']
                self.time = np.stack(
                    [self.week, self.month, self.year], axis=1)
                lon, lat = np.meshgrid(
                    (dataset.lon-180) * d2r, dataset.lat*d2r)
                x, y, z = latlon2xyz(lat, lon)
                self.V = np.stack([x, y, z]).reshape(3, self.shape[0]*self.shape[1]).T
            if not single_variant and isinstance(self.level, list):
                dataset = dataset.sel(level=np.array(self.level))
            data = dataset.get(data_name).values[:, np.newaxis, :, :]

        elif data_name == 'uv10':
            input_datasets = []
            for key in ['u10', 'v10']:
                try:
                    dataset = xr.open_mfdataset(self.data_root+'/{}/{}*.nc'.format(
                        data_map[key], data_map[key]), combine='by_coords')
                except (AttributeError, ValueError):
                    assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                     'pip install xarray==0.19.0,' \
                                     'pip install netcdf4 h5netcdf dask'
                except OSError:
                    print("OSError: Invalid path {}/{}/*.nc".format(self.data_root, data_map[key]))
                    assert False
                dataset = dataset.sel(time=slice(*self.training_time))
                dataset = dataset.isel(time=slice(None, -1, self.step))
                if self.time is None and single_variant:
                    self.week = dataset['time.week']
                    self.month = dataset['time.month']
                    self.year = dataset['time.year']
                    self.time = np.stack(
                        [self.week, self.month, self.year], axis=1)
                    lon, lat = np.meshgrid(
                        (dataset.lon-180) * d2r, dataset.lat*d2r)
                    x, y, z = latlon2xyz(lat, lon)
                    self.V = np.stack([x, y, z]).reshape(3, self.shape[0]*self.shape[1]).T
                input_datasets.append(dataset.get(key).values[:, np.newaxis, :, :])
            data = np.concatenate(input_datasets, axis=1)

        # uv10
        if len(data.shape) == 5:
            data = data.squeeze(1)
        # humidity
        if data_name == 'r' and single_variant:
            data = data[:, -1:, ...]
        # multi-variant level
        if not single_variant and isinstance(self.level, int):
            data = data[:, -self.level:, ...]

        mean = data.mean(axis=(0, 2, 3)).reshape(1, data.shape[1], 1, 1)
        std = data.std(axis=(0, 2, 3)).reshape(1, data.shape[1], 1, 1)
        # mean = dataset.mean('time').mean(('lat', 'lon')).compute()[data_name].values
        # std = dataset.std('time').mean(('lat', 'lon')).compute()[data_name].values
        data = (data - mean) / std

        return data, mean, std

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        seqs = F.interpolate(seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        index = self.valid_idx[index]
        data = torch.tensor(self.data[index+self.idx_in])
        labels = torch.tensor(self.data[index+self.idx_out])
        if self.use_augment:
            seqs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.96)
            return seqs
        else:
            return torch.cat((data, labels),dim=0)

class WeatherBench(pl.LightningDataModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(copy.deepcopy(hparams))
        assert hparams.data_split in ['5_625', '2_8125', '1_40625']
        _dataroot = osp.join(hparams.base_dir, f'weather_{hparams.data_split}deg')
        self.base_dir = _dataroot if osp.exists(_dataroot) else osp.join(hparams.base_dir, 'weather')


    @staticmethod
    def add_data_specific_args(
            parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--base_dir", type=str, default="models_pytorch/datasets/")
        parser.add_argument('--data_name_v',  type=str, default="t2m")
        parser.add_argument('--train_time', type=ast.literal_eval, default=['2010', '2015'])
        parser.add_argument('--val_time', type=ast.literal_eval, default=['2016', '2016'])
        parser.add_argument('--test_time', type=ast.literal_eval, default=['2017', '2018'])
        parser.add_argument('--data_split', type=str, default="5_625")
        parser.add_argument('--use_augment', type=str2bool, default=False)
        parser.add_argument('--idx_in', type=ast.literal_eval, default=[-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0])
        parser.add_argument('--idx_out', type=ast.literal_eval, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        parser.add_argument('--step', type=int, default=1)
        parser.add_argument('--level', type=int, default=1)
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument(
            "--num_workers", type=int, default=multiprocessing.cpu_count()
        )

        return parser


    def setup(self, stage: str =None):

        self.train_set = WeatherBenchDataset(data_root=self.base_dir,
                                        data_name=self.hparams.data_name_v, data_split=self.hparams.data_split,
                                        training_time=self.hparams.train_time,
                                        idx_in=self.hparams.idx_in,
                                        idx_out=self.hparams.idx_out,
                                        step=self.hparams.step, level=self.hparams.level,
                                        use_augment=self.hparams.use_augment)
        print(f"mean:{self.train_set.mean}, std: {self.train_set.std}")
        if stage == "fit" or None:
            self.val_set = WeatherBenchDataset(data_root=self.base_dir,
                                        data_name=self.hparams.data_name_v, data_split=self.hparams.data_split,
                                        training_time=self.hparams.test_time,
                                        idx_in=self.hparams.idx_in,
                                        idx_out=self.hparams.idx_out,
                                        step=self.hparams.step, level=self.hparams.level,use_augment=False,
                                        mean=self.train_set.mean,
                                        std=self.train_set.std)
            print(f"Train dataset: {len(self.train_set)} sequences, val dataset: {len(self.val_set)} sequences,")

        if stage == "test" or None:
            self.test_set = WeatherBenchDataset(data_root=self.base_dir,
                                               data_name=self.hparams.data_name_v, data_split=self.hparams.data_split,
                                               training_time=self.hparams.test_time,
                                               idx_in=self.hparams.idx_in,
                                               idx_out=self.hparams.idx_out,
                                               step=self.hparams.step, level=self.hparams.level, use_augment=False,
                                               mean=self.train_set.mean,
                                               std=self.train_set.std)
            print(f"Test dataset: {len(self.test_set)} sequences")


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True, pin_memory=True, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False, pin_memory=True, drop_last=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
