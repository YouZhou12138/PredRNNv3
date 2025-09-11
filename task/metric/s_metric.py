from task.metric.mse import MSE
from task.metric.mse_weather import MSE_w
from task.metric.rmse import RootMeanSquaredError
from task.metric.psnr import PSNR

METRICS = {"MSE":MSE,
           "RMSE": RootMeanSquaredError,
           "PSNR":PSNR,
           "MSE_w": MSE_w,
           }