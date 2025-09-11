from datasets.kth import KTH
from datasets.movingmnist import MovingMNIST
from datasets.taxibj import Taxibj
from datasets.weather import WeatherBench

DATASETS = {
    "moving_mnist": MovingMNIST,
    "kth": KTH,
    "taxibj": Taxibj,
    "weather": WeatherBench,
}


METRIC_CHECKPOINT_INFO = {
    "weather": {
        "monitor": "MSE",
        "filename": "{epoch:02d}-{MSE:.4f}",
        "mode": "min",
    },

    "moving_mnist": {
        "monitor": "MSE",
        "filename": "{epoch:02d}-{MSE:.4f}",
        "mode": "min",
    },

    "kth": {
        "monitor": "PSNR",
        "filename": "{epoch:02d}-{PSNR:.4f}",
        "mode": "max",
    },
    "taxibj": {
        "monitor": "MSE",
        "filename": "{epoch:02d}-{MSE:.4f}",
        "mode": "min",
    },
}