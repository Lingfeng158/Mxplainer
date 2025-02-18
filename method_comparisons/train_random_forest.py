from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import statistics
import torch.nn.functional as F
import torch
from tqdm import tqdm
import datetime
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score


validation_acc = 0.0
validation_loss = 100.0

torch.set_num_threads(2)

if __name__ == "__main__":
    # configs
    data_path = "/mnt/storage/llf/test/sl_data_featurefull3"
    meta_path = "/mnt/storage/llf/test/count_featurefull3.json"
    # data_path = "../data/test/sl_data_featurefull3"
    # meta_path = "../data/test/count_featurefull3.json"

    logdir_base = "logs"
    log_note = "test_featurefull3_random_forest"
    now = datetime.datetime.now()
    logdir = os.path.join(logdir_base, now.strftime("%m_%d_%H_%M") + log_note)
    sd_data_name = "/checkpoint/best_v_loss.pkl"
    sd_data_path = logdir + sd_data_name
    load_data = False  # if resume from previous training session
    writer = SummaryWriter(logdir)

    # prepare log dir
    if not os.path.exists(logdir + "/checkpoint"):
        os.makedirs(logdir + "/checkpoint")

    splitRatio = 0.93
    batchSize = 512
    epoch_total = 60
    n_gpu = 1

    # Load dataset
    training_DS = MahjongGBDataset(data_path, meta_path, 0, 0.92)
    test_DS = MahjongGBDataset(data_path, meta_path, 0.93, 1)

    X_train = []
    y_train = []
    for data, mask, target in training_DS:
        X_train.append(data.numpy().flatten())
        y_train.append(target)

    X_test = []
    y_test = []
    for data, mask, target in test_DS:
        X_test.append(data.numpy().flatten())
        y_test.append(target)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 创建随机森林分类器
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练模型
    rf.fit(X_train, y_train)
    # 预测测试集
    y_pred = rf.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # 预测每个样本属于各个类别的概率
    y_pred_proba = rf.predict_proba(X_test)

    # 获取每个样本概率最高的前 3 个类别索引
    prob_indices = np.argsort(y_pred_proba, axis=1)
    prob = np.sort(y_pred_proba, axis=1)
    top_3_indices = prob_indices[:, -3:]

    # 初始化正确预测计数
    correct_top_3 = 0

    # 遍历每个样本，检查真实标签是否在前 3 个预测类别中
    for i in range(len(y_test)):
        if y_test[i] in top_3_indices[i] or y_test[i] == y_pred[i]:
            correct_top_3 += 1

    # 计算 top-3 准确率
    top_3_accuracy = correct_top_3 / len(y_test)

    print(f"Top-3 Accuracy: {top_3_accuracy * 100:.2f}%")
