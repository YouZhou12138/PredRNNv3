import numpy as np
from torchmetrics import Metric
import torch
from skimage.metrics import structural_similarity as compare_ssim

class MSE_w(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(self, target_length: int,):

        super().__init__()
        # State variables for the metric
        self.add_state("sum_mse_error_t", default=torch.zeros(target_length), dist_reduce_fx="sum")
        self.add_state("sum_ssim_t", default=torch.zeros(target_length), dist_reduce_fx="sum")
        self.add_state("sum_mae_t", default=torch.zeros(target_length), dist_reduce_fx="sum")
        self.add_state("sum_rmse_t", default=torch.zeros(target_length), dist_reduce_fx="sum")
        self.add_state("sum_idx", default=torch.tensor(0), dist_reduce_fx="sum")
        self.mean = 278.78653
        self.std = 21.17883
        self.target_length = target_length


    def update(self, preds, targs, test=False):
        targs = targs[:, -self.target_length:]
        preds = preds[:, -self.target_length:]
        targs = targs * self.std + self.mean
        preds = preds * self.std + self.mean
        # The frame-by-frame error is calculated
        self.sum_idx += 1
        pixel_error = preds - targs

        mse_mean_t = torch.mean(pixel_error**2, dim=(0, 2, 3, 4))
        self.sum_mse_error_t += mse_mean_t

        if test:
            mae_mean_t = torch.mean(torch.abs(pixel_error), dim=(0, 2, 3, 4))
            self.sum_mae_t += mae_mean_t
            rmse_mean_t = torch.sqrt(mse_mean_t)
            self.sum_rmse_t += rmse_mean_t
            b, t, c, h, w = preds.shape
            multichannel = True if c > 1 else False
            preds_t = preds.reshape(-1, c, h, w)
            targs_t = targs.reshape(-1, c, h, w)
            # ssim_t and psnr
            if multichannel:
                real_frm = np.transpose(targs_t.cpu().numpy(), (0, 2, 3, 1))
                pred_frm = np.transpose(preds_t.cpu().numpy(), (0, 2, 3, 1))

            else:
                real_frm = targs_t.squeeze(1).cpu().numpy()
                pred_frm = preds_t.squeeze(1).cpu().numpy()

            ssim_t = torch.zeros(preds_t.shape[0], device=preds_t.device)
            for i in range(preds_t.shape[0]):
                score, _ = compare_ssim(pred_frm[i], real_frm[i], full=True,  multichannel=multichannel)
                ssim_t[i] += torch.tensor(score, device=preds_t.device)

            self.sum_ssim_t += ssim_t.view(b, t).mean(dim=0)


    def compute(self, validation=True,):
        if validation:
            return {"MSE": self.sum_mse_error_t.mean() / self.sum_idx}
        else:
            score = {"MSE": self.sum_mse_error_t.mean() / self.sum_idx,
                     "MAE": self.sum_mae_t.mean() / self.sum_idx,
                     "ssim": self.sum_ssim_t.mean() / self.sum_idx,
                     "rmse": self.sum_rmse_t.mean() / self.sum_idx,
                     }

            score_t = {"MSE_t": self.sum_mse_error_t / self.sum_idx,
                       "MAE_t": self.sum_mae_t / self.sum_idx,
                       "ssim_t": self.sum_ssim_t / self.sum_idx,
                       "rmse_t": self.sum_rmse_t / self.sum_idx,
                    }
            return score, score_t
