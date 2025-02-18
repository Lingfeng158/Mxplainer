from blackJack_DS import blackJackDataset
from blackJack_neural import neural_agent
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import statistics
from tqdm import tqdm
import datetime
import os
import sys
import json
import shutil

validation_acc = 0.0


def prepare_device(n_gpu_use, main_id=0):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    main_id for specify main gpu
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:%d" % main_id if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def loss_acc_func(pred, label):
    # label_action, label_tile = label_action.float(), label_tile.float()
    acc_count = 0
    mse_loss = F.mse_loss(pred, label)

    diff = abs(pred - label)
    # print(pred[:10], label[:10], diff[:10], sum(1 * (diff[:10] < 1)))
    total = len(pred)

    return (mse_loss, sum(1 * (diff < 0.4)), total)


def workload(
    network,
    trial_loader,
    epoch_it,
    device,
    optimizer,
    writer,
    log_dir,
    is_training=True,
):
    global validation_acc

    loader = trial_loader

    if is_training:
        network.train()
    else:
        network.eval()

    loss_list = []
    acc_count = 0
    acc_total = 0
    for i, d in enumerate(
        tqdm(loader, desc="Training: " if is_training else "Validating: ")
    ):
        data_a, data_b, label = d
        data_a, data_b, label = (data_a.to(device), data_b.to(device), label.to(device))

        if is_training:
            optimizer.zero_grad()
            pred = network((data_a, data_b))
            loss, acc, total = loss_acc_func(pred, label)
            loss.backward()
            loss_list.append(loss.item())
            acc_count += acc
            acc_total += total
            optimizer.step()
        else:
            with torch.no_grad():
                pred = network((data_a, data_b))
                loss, acc, total = loss_acc_func(pred, label)
                loss_list.append(loss.item())
                acc_count += acc
                acc_total += total

    acc = acc_count / acc_total * 100
    if is_training:
        torch.save(network.state_dict(), log_dir + "/checkpoint/%d.pkl" % epoch_it)
        writer.add_scalar(
            "loss_global/loss_train", statistics.mean(loss_list), epoch_it
        )
        writer.add_scalar("acc_global/acc_train", acc, epoch_it)
    else:
        writer.add_scalar(
            "loss_global/loss_validate", statistics.mean(loss_list), epoch_it
        )
        writer.add_scalar("acc_global/acc_validate", acc, epoch_it)
        # if acc > validation_acc:
        #     torch.save(network.state_dict(), log_dir + "/checkpoint/best.pkl")
        #     validation_acc = acc


if __name__ == "__main__":
    logdir = "log/"
    epoch_total = 15
    data_name = "checkpoint/best.pkl"
    torch.manual_seed(3407)

    now = datetime.datetime.now()
    run_summary_dir = os.path.join(logdir, "{}".format(now.strftime("%m_%d_%H_%M_%S")))
    writer = SummaryWriter(run_summary_dir)

    # prepare log dir
    if not os.path.exists(os.path.join(run_summary_dir, "checkpoint")):
        os.makedirs(os.path.join(run_summary_dir, "checkpoint"))

    device, device_ids = prepare_device(1, 0)

    # Load training dataset
    ds_training = blackJackDataset("./bj_opt_data.json")
    loader_training = DataLoader(
        dataset=ds_training, batch_size=128, shuffle=True, num_workers=4
    )

    # Load testing dataset
    ds_testing = blackJackDataset("./bj_opt_testing.json")
    Loader_testing = DataLoader(
        dataset=ds_testing, batch_size=128, shuffle=False, num_workers=4
    )

    nn = neural_agent(device).to(device)
    optimizer = torch.optim.Adam(nn.parameters(), lr=5e-2)

    # training
    for i in tqdm(range(0, epoch_total), desc="Epoch Progress: "):
        # training
        workload(
            nn,
            loader_training,
            i,
            device,
            optimizer,
            writer,
            run_summary_dir,
            True,
        )
        # validation
        workload(
            nn,
            Loader_testing,
            i,
            device,
            optimizer,
            writer,
            run_summary_dir,
            False,
        )
        writer.flush()

    writer.close()
