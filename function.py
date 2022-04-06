import random
from collections import OrderedDict
from os.path import join

import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.common import *
from lib.datasetV3 import TrainDatasetV3
from lib.extract_patches import get_data_train
from lib.losses.loss import *
from lib.metrics import Evaluate
from lib.visualize import group_images, save_img


# ========================get dataloader==============================
def get_dataloaderV3(path, args):
    """
    for spine dataset
    """
    train_set = TrainDatasetV3(path, mode="train", args=args)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=6
    )

    val_set = TrainDatasetV3(path, mode="val", args=args)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=6
    )

    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        visual_set = TrainDatasetV3(path, mode="val", args=args)
        visual_loader = DataLoader(
            visual_set, batch_size=1, shuffle=True, num_workers=0
        )
        N_sample = 50
        visual_imgs = np.empty(
            (N_sample, 1, args.train_patch_height, args.train_patch_width)
        )
        visual_masks = np.empty(
            (N_sample, 1, args.train_patch_height, args.train_patch_width)
        )

        for i, (img, mask) in tqdm(enumerate(visual_loader)):
            visual_imgs[i] = np.squeeze(img.numpy(), axis=0)
            visual_masks[i, 0] = np.squeeze(mask.numpy(), axis=0)
            if i >= N_sample - 1:
                break
        save_img(
            group_images((visual_imgs[0:N_sample, :, :, :] * 255).astype(np.uint8), 10),
            join(args.outf, args.save, "sample_input_imgs.png"),
        )
        save_img(
            group_images(
                (visual_masks[0:N_sample, :, :, :] * 255).astype(np.uint8), 10
            ),
            join(args.outf, args.save, "sample_input_masks.png"),
        )
    return train_loader, val_loader


# =======================train========================
def train(train_loader, net, criterion, optimizer, device, scaler=None):
    net.train()
    train_loss = AverageMeter()

    for batch_idx, (inputs, targets) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = net(inputs)
                output = torch.sigmoid(outputs)
                # output = output.view(output.size(0), -1).float()
                # target = targets.view(targets.size(0), -1).float()
                loss = criterion(output, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = net(inputs)
            output = torch.sigmoid(outputs)
            # output = output.view(output.size(0), -1).float()
            # target = targets.view(targets.size(0), -1).float()
            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
    log = OrderedDict([("train_loss", train_loss.avg)])
    return log


# ========================val===============================
def val(val_loader, net, criterion, device):
    net.eval()
    val_loss = AverageMeter()
    evaluater = Evaluate()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(
            enumerate(val_loader), total=len(val_loader)
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            output = torch.sigmoid(outputs)

            # output = output.view(output.size(0), -1).float()
            # target = targets.view(targets.size(0), -1).float()
            loss = criterion(output, targets)
            val_loss.update(loss.item(), inputs.size(0))

            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            evaluater.add_batch(targets, outputs)
    log = OrderedDict(
        [
            ("val_loss", val_loss.avg),
            ("val_acc", evaluater.confusion_matrix()[1]),
            ("val_f1", evaluater.f1_score()),
            ("val_auc_roc", evaluater.auc_roc()),
        ]
    )
    return log
