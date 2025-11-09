import torch
from dataProcess import process_data
from train import model_k_fold_train, model_test_train, predict_threshold
import os
import numpy as np
from model.model import CAFN
import random
from Metrics import Metrics,calculate_custom_metrics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Model = CAFN()

def load_model(model_path):
    model = Model.to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        return model
    else:
        print("[ERROR]\tModel not found at the specified path.")
        return None

def set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_randomseed(40)
    sample_Strategy = 3
    X_train, Y_train, label_train, X_test, Y_test, label_test, original_train_X, original_train_Y, original_train_label,\
        X_train2, original_train_X2, X_test2 = process_data(
        "dataset/Train dataset", "./dataset/Test dataset", sample_Strategy)

    original_train_X2 = torch.tensor(original_train_X2, dtype=torch.float32)
    X_train2 = torch.tensor(X_train2, dtype=torch.float32)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = Y_train
    Y_train_label = torch.tensor(label_train, dtype=torch.float32)
    X_ori = torch.tensor(original_train_X, dtype=torch.float32)
    Y_ori = torch.tensor(original_train_label, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_test2 = torch.tensor(X_test2, dtype=torch.float32)
    label_test = torch.tensor(label_test, dtype=torch.float32)

    model_k_fold_train(X_train=X_train, targets_train=Y_train_label, X_ori=X_ori, Y_ori=Y_ori, Y_label=Y_train, X_train2=X_train2, X_ori2=original_train_X2)
    model_test_train(X_train=X_train, targets_train=Y_train_label, Y_train=Y_train, X_train2=X_train2)
    file_list = os.listdir('ckpt')  # os.listdir('./ckpt')
    adam_files = [file_name for file_name in file_list if file_name.startswith('Adam')]
    sorted_adam_files = sorted(adam_files, key=lambda x: int(x.split('_epochs')[1].split('.pth')[0]))

    for file in sorted_adam_files:
        if file.startswith('Adam'):
            model_path = os.path.join('ckpt', file)
            model = load_model(model_path=model_path).to(device)
            predictions = predict_threshold(model, X_test, X_test2, label_test)
            test_calculator = Metrics()
            test_calculator.calculate_metrics(label_test.cpu(), predictions.cpu())
            test_calculator.transform_format()
            test_calculator.accumulate_counts(label_test.cpu(), predictions.cpu())
            ratio = test_calculator.calculate_each_class_absolute_true_rate()
            print(f"[INFO]{file}\teach class absolute true rate in test set:{ratio}")
            mrj = calculate_custom_metrics(label_test.cpu().numpy(), predictions.cpu().numpy())
            print(mrj)
