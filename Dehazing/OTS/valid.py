import os

import torch
import torch.nn.functional as f
from data import valid_dataloader
from pytorch_msssim import ssim as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision.transforms import functional as F
from utils import Adder


def _valid(model, args, ep, device):
    ots = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 4
        for idx, data in enumerate(ots):
            input_img, label_img = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)
            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = compare_psnr(p_numpy, label_numpy, data_range=1)
            ssim = compare_ssim(pred_clip, label_img, data_range=1, size_average=False).mean()


            psnr_adder(psnr)
            ssim_adder(ssim)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average(), ssim_adder.average()
