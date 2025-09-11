import numpy as np
from torchmetrics import Metric
import torch
from skimage.metrics import structural_similarity as compare_ssim

class PSNR(Metric):
    def __init__(self, target_length: int,):

        super().__init__()
        # State variables for the metric
        self.add_state("sum_psnr_t",  default=torch.zeros(target_length), dist_reduce_fx="sum")
        self.add_state("sum_ssim_t", default=torch.zeros(target_length), dist_reduce_fx="sum")
        self.add_state("sum_idx", default=torch.tensor(0), dist_reduce_fx="sum")
        self.target_length = target_length


    def update(self, preds, targs, test=False):
        targs = targs[:, -self.target_length:]
        preds = preds[:, -self.target_length:]
        preds = torch.clip(preds, min=0, max=1)

        self.sum_idx += 1
        b, t, c, h, w = preds.shape
        multichannel = True if c > 1 else False
        preds_t = preds.reshape(-1, c, h, w)
        targs_t = targs.reshape(-1, c, h, w)
        if multichannel:
            real_frm = np.transpose(targs_t.cpu().numpy(), (0, 2, 3, 1))
            pred_frm = np.transpose(preds_t.cpu().numpy(), (0, 2, 3, 1))

        else:
            real_frm = targs_t.squeeze(1).cpu().numpy()
            pred_frm = preds_t.squeeze(1).cpu().numpy()
        psnr_t = torch.zeros(preds_t.shape[0], device=preds_t.device)

        for i in range(preds_t.shape[0]):
           psnr_t[i] += torch.tensor(self.PSNR(pred=pred_frm[i], true=real_frm[i]),device=preds_t.device)
        self.sum_psnr_t += psnr_t.view(b, t).mean(dim=0)

        if test:
            ssim_t = torch.zeros(preds_t.shape[0], device=preds_t.device)
            for i in range(preds_t.shape[0]):
                score, _ = compare_ssim(pred_frm[i], real_frm[i], full=True, multichannel=multichannel)
                ssim_t[i] += torch.tensor(score, device=preds_t.device)

            self.sum_ssim_t += ssim_t.view(b, t).mean(dim=0)


    def PSNR(self, pred, true):
        mse = np.mean((np.uint8(pred * 255) - np.uint8(true * 255)) ** 2)
        return 20 * np.log10(255) - 10 * np.log10(mse)

    def compute(self, validation=True, ):
        if validation:
            return {"PSNR": self.sum_psnr_t.mean() / self.sum_idx, }
        else:
            score = {
                     "ssim": self.sum_ssim_t.mean() / self.sum_idx,
                     "psnr": self.sum_psnr_t.mean() / self.sum_idx,
                     }

            score_t = {"ssim_t": self.sum_ssim_t / self.sum_idx,
                       "psnr_t": self.sum_psnr_t / self.sum_idx,
                       }
            return score, score_t
