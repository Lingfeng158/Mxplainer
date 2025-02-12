from dataset import MahjongSLDataset
from model_slim_rev1 import MahJongNetBatchedRevised
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


# generate negative case from certain saved weight-file
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
    info_input=None,
    data=None,
):
    # label_action, label_tile = label_action.float(), label_tile.float()
    (
        meta_feature_new,
        tile_wall_feature,
        (tile_prep, fan_prep, missing_tile_prep, count_prep, chi_peng_count_remain),
    ) = data
    meta_as_index = index_list_plus_one * meta
    reverse_meta_as_index = index_list_plus_one * (1 - meta)
    meta_as_index = meta_as_index[meta_as_index > 0] - 1
    reverse_meta_as_index = reverse_meta_as_index[reverse_meta_as_index > 0] - 1
    acc1 = sum(
        action[meta_as_index].argmax(dim=1) == label_action[meta_as_index].argmax(dim=1)
    )

    # compose info for negative cases
    info_dict = {
        "data": [],
        "pred": [],
        "label": [],
        "info": [],
        "negative_action_acc": [],
        "negative_action_total": [],
        "negative_tile_acc": [],
        "negative_tile_total": [],
    }
    acc1_total = len(meta_as_index)
    acc1_with_action = sum(
        (
            action[meta_as_index].argmax(dim=1)
            == label_action[meta_as_index].argmax(dim=1)
        )
        & (action[meta_as_index].argmax(dim=1) != 0)
    )
    acc1_with_action_total = sum((label_action[meta_as_index].argmax(dim=1) != 0))

    for id in meta_as_index:
        if action[id].argmax() != label_action[id].argmax():
            info_dict["data"].append(
                (
                    meta_feature_new[id].cpu().numpy().tolist(),
                    tile_wall_feature[id].cpu().numpy().tolist(),
                    tile_prep[id].cpu().numpy().tolist(),
                    fan_prep[id].cpu().numpy().tolist(),
                    missing_tile_prep[id].cpu().numpy().tolist(),
                    count_prep[id].cpu().numpy().tolist(),
                    chi_peng_count_remain[id].cpu().numpy().tolist(),
                )
            )
            info_dict["pred"].append((action[id].detach().cpu().numpy().tolist()))
            info_dict["label"].append((label_action[id].cpu().numpy().tolist()))
            info_dict["info"].append((info_input[id].numpy().tolist()))
    info_dict["negative_action_acc"].append(acc1.item())
    info_dict["negative_action_total"].append(acc1_total)

    truth_label_index = label_tile[reverse_meta_as_index].argmax(dim=1)
    acc2 = sum(tile_sel[reverse_meta_as_index].argmax(dim=1) == truth_label_index)
    acc2_total = len(reverse_meta_as_index)

    # compose info for negative cases
    for id in reverse_meta_as_index:
        if tile_sel[id].argmax() != label_tile[id].argmax():
            info_dict["data"].append(
                (
                    (
                        meta_feature_new[id].cpu().numpy().tolist(),
                        tile_wall_feature[id].cpu().numpy().tolist(),
                        tile_prep[id].cpu().numpy().tolist(),
                        fan_prep[id].cpu().numpy().tolist(),
                        missing_tile_prep[id].cpu().numpy().tolist(),
                        count_prep[id].cpu().numpy().tolist(),
                        chi_peng_count_remain[id].cpu().numpy().tolist(),
                    )
                )
            )
            info_dict["pred"].append((tile_sel[id].detach().cpu().numpy().tolist()))
            info_dict["label"].append((label_tile[id].cpu().numpy().tolist()))
            info_dict["info"].append((info_input[id].numpy().tolist()))
    info_dict["negative_tile_acc"].append(acc2.item())
    info_dict["negative_tile_total"].append(acc2_total)

    # regular1 = 1e-3 * torch.sum((fan_coeff_param  - torch.abs(fan_coeff_param)) ** 2)# + 1e-4 * torch.sum((fan_coeff_param  - 1) ** 2)
    # regular2 = 1e-3 * torch.sum((tile_coeff_param - torch.abs(tile_coeff_param)) ** 2)# + 1e-4 * torch.sum((tile_coeff_param  - 1) ** 2)
    return (
        (
            acc1,
            acc2,
            acc1_total,
            acc2_total,
            acc1_with_action,
            acc1_with_action_total,
        ),
        info_dict,
    )


def workload(network, device, validation_loader, path_to_output, total_sample_size):
    """
    iteration:                  iteration in epoch
    begin/end_ratio:               training/validation set's range (0.0-1.0)
    persistent_validation_DS:   make validation set persistent in RAM,
                                which consumes more ram, but speeds up validation
                                set to True and is_training set to False, will nullify begin/end ratio
    validation_DS:              pre-loaded validation dataset
    validation_loader:          pre-loaded validation loader
    """
    info_dict = {
        "data": [],
        "pred": [],
        "label": [],
        "info": [],
        "negative_action_acc": [],
        "negative_action_total": [],
        "negative_tile_acc": [],
        "negative_tile_total": [],
    }
    network.eval()
    loader = validation_loader
    action_acc = 0
    action_total = 0
    tile_acc = 0
    tile_total = 0
    action_acc_do = 0
    action_acc_do_total = 0
    for i, d in enumerate(tqdm(loader, leave=False, desc="Validating: ")):
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

        x = (
            meta_feature_new,
            tile_wall_feature,
            data1,
        )

        with torch.no_grad():
            action, tile_choice = network(x)
            acc, negative_dict = loss_acc_batched(
                meta,
                index_list_plus_one,
                label_action,
                label_tile,
                action,
                tile_choice,
                info_input=v_info,
                data=(meta_feature_new, tile_wall_feature, data1),
            )
            (
                action_ac,
                tile_ac,
                action_tot,
                tile_tot,
                action_ac_do,
                action_ac_do_tot,
            ) = acc
            action_acc += action_ac
            tile_acc += tile_ac
            action_total += action_tot
            tile_total += tile_tot
            action_acc_do += action_ac_do
            action_acc_do_total += action_ac_do_tot
            # compose info for negative cases
            info_dict["data"].extend(negative_dict["data"])
            info_dict["pred"].extend(negative_dict["pred"])
            info_dict["label"].extend(negative_dict["label"])
            info_dict["info"].extend(negative_dict["info"])
            info_dict["negative_action_acc"].extend(
                negative_dict["negative_action_acc"]
            )
            info_dict["negative_action_total"].extend(
                negative_dict["negative_action_total"]
            )
            info_dict["negative_tile_acc"].extend(negative_dict["negative_tile_acc"])
            info_dict["negative_tile_total"].extend(
                negative_dict["negative_tile_total"]
            )

    # info_dict["negative_ct"] = info_dict["negative_ct"].item()
    acc = (tile_acc + action_acc) / (tile_total + action_total) * 100
    info_dict["acc"] = acc.item()
    info_dict["total"] = total_sample_size
    with open(path_to_output, "w") as f:
        json.dump(
            {
                "negative_dict": info_dict,
            },
            f,
        )


if __name__ == "__main__":
    json_name = "neg_slim.json"
    output_path = "./"
    main_gpu_id = 0
    data_folder_name = "sl_prep_slim_v6"
    log_dir_name = "log/07_01_19_29_35-model_slim_revD/checkpoint"
    # log_dir_name = "log/06_08_20_09_46-TFx_OB_v6_default/checkpoint"
    # log_dir_name = "./"

    # prepare log dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    device, device_ids = prepare_device(1, main_gpu_id)

    path_to_saved_weight = os.path.join(log_dir_name, "best_acc.pkl")
    path_to_output = os.path.join(output_path, json_name)

    # Pre-load validation dataset
    vDS = MahjongSLDataset(data_folder_name, 0.99, 1)
    total_sample_size = len(vDS)
    vloader = DataLoader(dataset=vDS, batch_size=256, shuffle=False, num_workers=4)
    nn = MahJongNetBatchedRevised(device).to(device)
    # nn.load_state_dict(
    #     torch.load(
    #         path_to_saved_weight,
    #         map_location=torch.device(device),
    #     )
    # )

    workload(nn, device, vloader, path_to_output, total_sample_size)
