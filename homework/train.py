# accuracy + TensorBoard + auto log + auto zip
import os
from .planner import Planner, save_model
import torch
import torch.utils.tensorboard as tb
from torchmetrics import Accuracy  # accuracy
import numpy as np
from .utils import load_data
from . import dense_transforms
import pdb
import shutil

import faulthandler
faulthandler.enable()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

basePath = "./homework/"


def train(args):
    from os import path
    print("Initializing Model =================================")
    model = Planner()
    train_logger, valid_logger = None, None

    writer = tb.SummaryWriter()  # TensorBoard

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    
    """
    print("installing torch ...")
    import torch

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), 'planner.th')))  # change load weight file name

    # auto log
    else:
        if (os.path.exists(basePath + "train_log.txt")):
            os.remove(basePath + "train_log.txt")
    # auto log

    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("installing inspect ...")
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(
        dense_transforms) if inspect.isclass(v)})
    
    print("loading data ...")
    train_data = load_data('drive_data', transform=transform,
                           num_workers=args.num_workers)  # change data set file name
    # for i, (a,b) in enumerate(train_data):
    #     ashape = a.shape
    #     bshape = b.shape
    #     pdb.set_trace()
    #     break

    global_step = 0
    print("Begin Training =================================")

    # accuracy
    max_acc = 0
    min_loss = 1
    # auto log
    start_epoch = 0
    if (os.path.exists(basePath + "train_log.txt")):
        with open(basePath + "train_log.txt", 'rb') as f:
            try:  # catch OSError in case of a one line file
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
            start_epoch = int(last_line.split(
                "_")[0].split("=")[1].strip()) + 1
            max_acc = float(last_line.split("_")[4].split("=")[1].strip())
            min_loss = float(last_line.split("_")[3].split("=")[1].strip())
    # auto log
    hasNewMax = False
    # accuracy

    for epoch in range(args.num_epoch):
        epoch += start_epoch  # auto log
        model.train()
        losses = []
        accuracies = []
        # pdb.set_trace()
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            pred = model(img)
            loss_val = loss(pred, label)
            # pdb.set_trace()

            # accuracy
            accuracy = Accuracy(mdmc_average='samplewise')
            p = torch.abs(torch.mul(pred, 10))
            l = torch.abs(torch.mul(label, 10))
            acc_val = accuracy(p.type(dtype=torch.int),
                               l.type(dtype=torch.int))
            # accuracy

            # TensorBoard
            writer.add_scalar('loss', loss_val, global_step)
            log(writer, img, label, pred, global_step)
            # TensorBoard

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                if global_step % 100 == 0:
                    log(train_logger, img, label, pred, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

            accuracies.append(acc_val)  # accuracy
            losses.append(loss_val.detach().cpu().numpy())

        avg_batch_acc = np.mean(accuracies)  # accuracy
        avg_batch_loss = np.mean(losses)

        # accuracy
        if (avg_batch_loss < min_loss):
            min_loss = avg_batch_loss
        if (avg_batch_acc > max_acc):
            max_acc = avg_batch_acc
            hasNewMax = True
        # accuracy

        # pdb.set_trace()

        logStr = 'epoch=' + str(epoch) + '_Loss=' + str(avg_batch_loss) + '_Acc=' + \
            str(avg_batch_acc) + '_MinLoss=' + \
            str(min_loss) + '_MaxAcc=' + str(max_acc) + "\n"
        with open(basePath + "train_log.txt", "a") as f:
            f.write(logStr)

        if train_logger is None:
            print('epoch %-3d \t Loss = %0.3f \t Acc = %0.3f  \t MinLoss = %0.3f \t MaxAcc = %0.3f' %
                  (epoch, avg_batch_loss, avg_batch_acc, min_loss, max_acc))

        save_model(model)

        # accuracy
        if (hasNewMax):  # store the highest accuracy as best weights
            try:
                src = path.join(basePath + 'planner.th')
                dst = path.join(basePath + 'planner_best.th')
                shutil.copy(src, dst)
            except:
                pass
        # accuracy

    save_model(model)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(
        WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(
        WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=10)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument(
        '-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
