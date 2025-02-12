from dataset import MahjongSLDataset
from model_slim_revB import MahJongNetBatchedRevised
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, ExponentialLR
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


def loss_acc_batched(
    meta,
    index_list_plus_one,
    label_action,
    label_tile,
    action,
    tile_sel,
):
    # label_action, label_tile = label_action.float(), label_tile.float()

    meta_as_index = index_list_plus_one * meta
    reverse_meta_as_index = index_list_plus_one * (1 - meta)
    meta_as_index = meta_as_index[meta_as_index > 0] - 1
    reverse_meta_as_index = reverse_meta_as_index[reverse_meta_as_index > 0] - 1
    term1 = F.cross_entropy(action[meta_as_index], label_action[meta_as_index]) * len(
        meta_as_index
    )
    acc1 = sum(
        action[meta_as_index].argmax(dim=1) == label_action[meta_as_index].argmax(dim=1)
    )
    acc1_total = len(meta_as_index)
    acc1_with_action = sum(
        (
            action[meta_as_index].argmax(dim=1)
            == label_action[meta_as_index].argmax(dim=1)
        )
        & (action[meta_as_index].argmax(dim=1) != 0)
    )
    acc1_with_action_total = sum((label_action[meta_as_index].argmax(dim=1) != 0))
    term2 = F.cross_entropy(
        tile_sel[reverse_meta_as_index], label_tile[reverse_meta_as_index]
    ) * len(reverse_meta_as_index)
    truth_label_index = label_tile[reverse_meta_as_index].argmax(dim=1)
    acc2 = sum(tile_sel[reverse_meta_as_index].argmax(dim=1) == truth_label_index)
    _, top3_index = torch.topk(tile_sel[reverse_meta_as_index], 3)
    acc2_top3 = 0
    for id in range(0, 3):
        acc2_top3 += sum(top3_index[:, id] == truth_label_index)
    acc2_total = len(reverse_meta_as_index)

    # regular1 = 1e-3 * torch.sum((fan_coeff_param  - torch.abs(fan_coeff_param)) ** 2)# + 1e-4 * torch.sum((fan_coeff_param  - 1) ** 2)
    # regular2 = 1e-3 * torch.sum((tile_coeff_param - torch.abs(tile_coeff_param)) ** 2)# + 1e-4 * torch.sum((tile_coeff_param  - 1) ** 2)

    return (1 * term1 + term2) / len(meta), (
        acc1,
        acc2,
        acc1_total,
        acc2_total,
        acc1_with_action,
        acc1_with_action_total,
        acc2_top3,
    )


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
    """
    iteration:                  iteration in epoch
    begin/end_ratio:               training/validation set's range (0.0-1.0)
    persistent_validation_DS:   make validation set persistent in RAM,
                                which consumes more ram, but speeds up validation
                                set to True and is_training set to False, will nullify begin/end ratio
    validation_DS:              pre-loaded validation dataset
    validation_loader:          pre-loaded validation loader
    """
    global validation_acc

    loader = trial_loader

    if is_training:
        network.train()
    else:
        network.eval()

    loss_list = []
    action_acc = 0
    action_total = 0
    tile_acc = 0
    tile_total = 0
    action_acc_do = 0
    action_acc_do_total = 0
    tile_acc_top3 = 0
    for i, d in enumerate(
        tqdm(loader, leave=False, desc="Training: " if is_training else "Validating: ")
    ):
        (
            meta,
            meta_feature_new,
            tile_wall_feature,
            k,
            q,
            m,
            n,
            o,
            # fan_summary,
            label_action,
            label_tile,
            v_info,
        ) = d

        (
            meta,
            meta_feature_new,
            tile_wall_feature,
            k,
            q,
            m,
            n,
            o,
            # fan_summary,
            label_action,
            label_tile,
        ) = (
            meta.to(device),
            meta_feature_new.to(device),
            tile_wall_feature.to(device),
            k.to(device),
            q.to(device),
            m.to(device),
            n.to(device),
            o.to(device),
            # data_description.to(device).float(),
            # fan_summary.to(device),
            label_action.to(device),
            label_tile.to(device),
        )
        data1 = (k, q, m, n, o)

        index_list_plus_one = torch.tensor([i + 1 for i in range(len(meta))]).to(device)
        x = (meta_feature_new, tile_wall_feature, data1)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            action, tile_choice = network(x)
            loss, acc = loss_acc_batched(
                meta,
                index_list_plus_one,
                label_action,
                label_tile,
                action,
                tile_choice,
            )
            loss.backward()
            loss_list.append(loss.item())
            (
                action_ac,
                tile_ac,
                action_tot,
                tile_tot,
                action_ac_do,
                action_ac_do_tot,
                tile_ac_top3,
            ) = acc
            action_acc += action_ac
            tile_acc += tile_ac
            action_total += action_tot
            tile_total += tile_tot
            action_acc_do += action_ac_do
            action_acc_do_total += action_ac_do_tot
            optimizer.step()
        else:
            with torch.no_grad():
                action, tile_choice = network(x)
                loss, acc = loss_acc_batched(
                    meta,
                    index_list_plus_one,
                    label_action,
                    label_tile,
                    action,
                    tile_choice,
                )
                loss_list.append(loss.item())
                (
                    action_ac,
                    tile_ac,
                    action_tot,
                    tile_tot,
                    action_ac_do,
                    action_ac_do_tot,
                    tile_ac_top3,
                ) = acc
                action_acc += action_ac
                tile_acc += tile_ac
                action_total += action_tot
                tile_total += tile_tot
                action_acc_do += action_ac_do
                action_acc_do_total += action_ac_do_tot
                tile_acc_top3 += tile_ac_top3

    if is_training:
        acc = (tile_acc + action_acc) / (tile_total + action_total) * 100
        acc_tile = tile_acc / tile_total * 100
        acc_action = action_acc / action_total * 100
        # print(acc, statistics.mean(loss_list))
        torch.save(network.state_dict(), log_dir + "/checkpoint/%d.pkl" % epoch_it)
        writer.add_scalar(
            "loss_global/loss_train", statistics.mean(loss_list), epoch_it
        )
        writer.add_scalar("acc_global/acc_train", acc, epoch_it)
        # writer.add_scalar("acc_global/tile_acc_train", acc_tile, epoch_it)
        # writer.add_scalar("acc_global/action_acc_train", acc_action, epoch_it)
    else:
        acc = (tile_acc + action_acc) / (tile_total + action_total) * 100
        acc_tile = tile_acc / tile_total * 100
        acc_tile_top3 = tile_acc_top3 / tile_total * 100
        acc_action = action_acc / action_total * 100
        active_action = action_acc_do / action_acc_do_total * 100
        writer.add_scalar(
            "loss_global/loss_validate", statistics.mean(loss_list), epoch_it
        )
        writer.add_scalar("acc_global/acc_validate", acc, epoch_it)
        writer.add_scalar("acc_global/tile_acc_validate", acc_tile, epoch_it)
        writer.add_scalar("acc_global/tile_acc_top3_validate", acc_tile_top3, epoch_it)
        writer.add_scalar("acc_global/action_acc_validate", acc_action, epoch_it)
        writer.add_scalar(
            "acc_global/active_action_acc_validate", active_action, epoch_it
        )
        if acc > validation_acc:
            torch.save(network.state_dict(), log_dir + "/checkpoint/best_acc.pkl")
            validation_acc = acc


if __name__ == "__main__":
    config_dir = "supervised_learning_v4/configs/"
    logdir = "log/"
    data_name = "checkpoint/best_v_loss.pkl"
    torch.manual_seed(3407)

    if len(sys.argv) > 1:
        config_file_path = os.path.join(config_dir, sys.argv[1])
        with open(config_file_path) as f:
            config = json.load(f)
    else:
        print("Requires config file for training")
        print("Using Default config")
        config = {
            "target_lr": 0.008,
            "lr_decay": 0.97,
            "splitRatio": 0.95,
            "batchSize": 256,
            "epoch_total": 2,
            "n_gpu": 1,
            "main_gpu_id": 0,
            "number_warmup_epochs": 4,
            "num_dataloader": 4,
            "data_folder_name": "sl_prep_revised6_QQR",
            "note": "slim_v8",
        }
        print(config)

    target_lr = config["target_lr"]
    lr_decay = config["lr_decay"]
    splitRatio = config["splitRatio"]
    batchSize = config["batchSize"]
    epoch_total = config["epoch_total"]
    n_gpu = config["n_gpu"]
    main_gpu_id = config["main_gpu_id"]
    number_warmup_epochs = config["number_warmup_epochs"]
    num_dataloader = config["num_dataloader"]
    data_folder_name = config["data_folder_name"]
    note = config["note"]

    now = datetime.datetime.now()
    run_summary_dir = os.path.join(
        logdir, "{}-{}".format(now.strftime("%m_%d_%H_%M_%S"), note)
    )
    writer = SummaryWriter(run_summary_dir)
    if len(sys.argv) > 1:
        shutil.copy(config_file_path, run_summary_dir)

    # prepare log dir
    if not os.path.exists(os.path.join(run_summary_dir, "checkpoint")):
        os.makedirs(os.path.join(run_summary_dir, "checkpoint"))

    device, device_ids = prepare_device(n_gpu, main_gpu_id)

    # Load training dataset
    tDS = MahjongSLDataset(data_folder_name, 0, splitRatio)
    tloader = DataLoader(
        dataset=tDS, batch_size=batchSize, shuffle=True, num_workers=num_dataloader
    )

    # Pre-load validation dataset
    vDS = MahjongSLDataset(data_folder_name, splitRatio, 1)
    vloader = DataLoader(
        dataset=vDS, batch_size=batchSize, shuffle=False, num_workers=num_dataloader
    )
    nn = MahJongNetBatchedRevised(device).to(device)
    optimizer = torch.optim.Adam(nn.parameters(), lr=target_lr)

    train_scheduler = ExponentialLR(optimizer, gamma=lr_decay)

    def warmup(current_step: int):
        return 1 / (2 ** (float(number_warmup_epochs - current_step)))

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)

    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, train_scheduler], [number_warmup_epochs]
    )

    # training
    for i in tqdm(range(0, epoch_total), desc="Epoch Progress: "):
        # training
        workload(
            nn,
            tloader,
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
            vloader,
            i,
            device,
            optimizer,
            writer,
            run_summary_dir,
            False,
        )
        scheduler.step()
        writer.flush()

    writer.close()
