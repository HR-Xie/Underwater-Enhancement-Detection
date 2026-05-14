import os
import warnings

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import torch
import torch.optim as optim
from accelerate import Accelerator, DistributedDataParallelKwargs

from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_absolute_error
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from config import Config
from data import get_training_data, get_test_data
from models import *
from utils import seed_everything, save_checkpoint

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)


def train():
    # Accelerate
    kwargs = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator(kwargs_handlers=kwargs)
    config = {
        "dataset": opt.TRAINING.TRAIN_DIR,
        "model": opt.MODEL.SESSION
    }
    accelerator.init_trackers("uw", config=config)

    criterion_psnr = torch.nn.MSELoss()

    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)

    # Data Loader
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    train_dataset = get_training_data(train_dir, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=8,
                             drop_last=False, pin_memory=True)
    val_dataset = get_test_data(val_dir, {'w': opt.TESTING.PS_W, 'h': opt.TESTING.PS_H, 'ori': opt.TRAINING.ORI})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

    # Initialize LightAquaNet
    model = LightAquaNet(use_prior=True, use_lka=True, use_gate=True)

    optimizer = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL,
                            betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    start_epoch = 1
    best_psnr = 0

    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0

        for i, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            # get the inputs; data is a list of [target, input, filename]
            tar = data[1]
            inp = data[0].contiguous()

            # forward
            optimizer.zero_grad()
            res = model(inp)

            loss_psnr = sum([criterion_psnr(res[j], tar) for j in range(len(res))])
            loss_ssim = sum(
                [(1 - structural_similarity_index_measure(res[j], tar, data_range=1)) for j in range(len(res))])

            train_loss = loss_psnr + 0.4 * loss_ssim

            # backward
            accelerator.backward(train_loss)
            optimizer.step()

        scheduler.step()

        if accelerator.is_local_main_process:
            print("epoch: {}, Loss: {}".format(epoch, train_loss))

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            with torch.no_grad():
                psnr = 0
                ssim = 0
                stat_lpips = 0
                stat_mae = 0
                for _, test_data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                    tar = test_data[1]
                    inp = test_data[0].contiguous()

                    res = model(inp)[0].clamp(0, 1)

                    psnr += peak_signal_noise_ratio(res, tar, data_range=1)
                    ssim += structural_similarity_index_measure(res, tar, data_range=1)
                    stat_mae += mean_absolute_error(torch.mul(res, 255), torch.mul(tar, 255))
                    stat_lpips += criterion_lpips(res, tar).item()

                psnr /= len(testloader)
                ssim /= len(testloader)
                stat_mae /= len(testloader)
                stat_lpips /= len(testloader)

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_ssmi = ssim
                    best_mae = stat_mae
                    best_lpips = stat_lpips
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch, opt.TRAINING.SAVE_DIR, opt.MODEL.SESSION)

                if accelerator.is_local_main_process:
                    accelerator.log({
                        "PSNR": psnr,
                        "SSIM": ssim
                    }, step=epoch)
                    print(
                        "epoch: {}, PSNR: {}, SSIM: {}, ||best PSNR: {},SSMI: {},MAE: {},LPIPS: {}".format(epoch, psnr,
                                                                                                           ssim,
                                                                                                           best_psnr,
                                                                                                           best_ssmi,
                                                                                                           best_mae,
                                                                                                           best_lpips))

    accelerator.end_training()


if __name__ == '__main__':
    train()
