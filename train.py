import sys
import time
from os.path import join
from test import Test

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler

import models
from config import parse_args
from function import get_dataloaderV3, train, val
from lib.common import *
from lib.logger import Logger, Print_Logger
from lib.losses.loss import *


def main(use_amp=True):
    setpu_seed(2021)
    args = parse_args()
    save_path = join(args.outf, args.save)
    save_args(args, save_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True

    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path, "train_log.txt"))
    print("The computing device used is: ", "GPU" if device.type == "cuda" else "CPU")

    net = models.GT_UNet.GT_U_Net(1, 1).to(device)

    ngpu = 1
    if ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(ngpu)))
    net = net.to(device)

    print("Total number of parameters: " + str(count_parameters(net)))

    log.save_graph(
        net, torch.randn((1, 1, 64, 64)).to(device).to(device=device)
    )  # Save the model structure to the tensorboard file
    # torch.nn.init.kaiming_normal(net, mode='fan_out')      # Modify default initialization method
    # net.apply(weight_init)

    # criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))
    criterion = torch.nn.CrossEntropyLoss()  # Initialize loss function
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # create a list of learning rate with epochs
    # lr_schedule = make_lr_schedule(np.array([50, args.N_epochs]),np.array([0.001, 0.0001]))
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.N_epochs, eta_min=0
    )

    if use_amp is True:
        scaler = GradScaler()
    else:
        scaler = None

    # The training speed of this task is fast, so pre training is not recommended
    if args.pre_trained is not None:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        checkpoint = torch.load(args.outf + "%s/latest_model.pth" % args.pre_trained)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1

    train_loader, val_loader = get_dataloaderV3(
        "./data_path_list/dataset.txt", args
    )  # create dataloader

    if args.val_on_test:
        print("\033[0;32m===============Validation on Testset!!!===============\033[0m")
        val_tool = Test(args)
    else:
        val_tool = None

    best = {
        "epoch": 0,
        "AUC_roc": 0.5,
    }  # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # Early stop Counter
    for epoch in range(args.start_epoch, args.N_epochs + 1):
        print(
            "\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s"
            % (
                epoch,
                args.N_epochs,
                optimizer.state_dict()["param_groups"][0]["lr"],
                time.asctime(),
            )
        )

        # train stage
        train_log = train(
            train_loader, net, criterion, optimizer, device, scaler=scaler
        )
        # val stage
        if args.val_on_test and val_tool is not None:
            val_tool.inference(net)
            val_log = val_tool.val()
        else:
            val_log = val(val_loader, net, criterion, device)

        log.update(epoch, train_log, val_log)  # Add log information
        lr_scheduler.step()

        # Save checkpoint of latest and best model.
        state = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(state, join(save_path, "latest_model.pth"))
        trigger += 1
        if val_log["val_auc_roc"] > best["AUC_roc"]:
            print("\033[0;33mSaving best model!\033[0m")
            torch.save(state, join(save_path, "best_model.pth"))
            best["epoch"] = epoch
            best["AUC_roc"] = val_log["val_auc_roc"]
            trigger = 0
        print(
            "Best performance at Epoch: {} | AUC_roc: {}".format(
                best["epoch"], best["AUC_roc"]
            )
        )
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
