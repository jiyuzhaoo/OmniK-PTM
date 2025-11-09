import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from Metrics import Metrics
import pandas as pd
from model.model import CAFN
import os
import warnings
warnings.filterwarnings('ignore')

B_S = 256
lr = 0.0001
num_epoch = 5000
weight_decay = 0.0001

test_num_epoch = 5000
test_lr = 0.0001
test_weight_decay = 0.0001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO]\tUsing device: {device}")


def predict_threshold(model, inputs, inputs2, targets, threshold=0.5):
    model.eval()
    inputs = inputs.to(device)
    inputs2 = inputs2.to(device)
    with torch.no_grad():
        test_outputs = model(inputs, inputs2)
        probs = torch.sigmoid(test_outputs)

        probs = probs.cpu().numpy()
        binary_preds = (probs >= [0.5, 0.5, 0.5, 0.5])
        binary_preds = torch.from_numpy(binary_preds).float()
    return binary_preds.to(device)

def save_model(model, save_path, num_epoch, is_train=True, fold=None):
    os.makedirs(save_path, exist_ok=True)
    if is_train:
        torch.save(model.state_dict(), os.path.join(save_path, f'{fold}fold_Adam_lr{lr}_weightdecay{weight_decay}_epochs{num_epoch}.pth'))
    else:
        torch.save(model.state_dict(),
                   os.path.join(save_path, f'Adam_lr{test_lr}_weightdecay{test_weight_decay}_epochs{num_epoch}.pth'))

def train_for_val(model, train_loader, X_train, X_train2, targets_train, X_val,
                  targets_val, optimizer, num_epochs, train_calculator, scheduler, fold, X_val2, is_train=True): # <- 在这里增加了 scheduler
    loss_Fn = nn.BCEWithLogitsLoss(reduction='mean')
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    best_loss = 11.0
    M_model = model
    for epoch in range(num_epochs):
        model.train()
        for batch_inputs, batch_inputs2, batch_targets, batch_y in train_loader:

            loss_fn = loss_Fn
            batch_inputs = batch_inputs.to(device)
            batch_inputs2 = batch_inputs2.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs, batch_inputs2)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train, X_train2)
                targets_train = targets_train.to(device)
                train_loss = loss_Fn(train_outputs, targets_train)

                val_outputs = model(X_val, X_val2)
                targets_val = targets_val.to(device)
                val_loss = loss_Fn(val_outputs, targets_val)
                print(f"[INFO]\tEpoch {epoch + 1}\t Train Loss: {train_loss.item():.4f}\tVal Loss: {val_loss.item():.4f}")
            scheduler.step(val_loss)
            #scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                save_model(model=model, save_path='ckpt', is_train=is_train, fold=fold, num_epoch=num_epochs)
                M_model = model

    train_predictions = predict_threshold(M_model, X_val, X_val2, targets_val)
    train_calculator.calculate_metrics(targets_val.cpu(), train_predictions.cpu())
    train_calculator.accumulate_counts(targets_val.cpu(), train_predictions.cpu())

def train_in_k_fold(X_train, Y_train, X_ori, Y_ori, Y_label, lr, weight_decay, num_epochs, X_train2, X_ori2, k=5, batch_size=B_S):
    kf = KFold(n_splits=k, shuffle=True, random_state=10)
    fold = 0
    train_calculator = Metrics()
    for train_idx, val_idx in kf.split(X_train):
        fold += 1
        print(f"[INFO]\tTraining on fold {fold}")

        X_train_fold = X_train[train_idx]
        X_train2_fold = X_train2[train_idx]
        Y_train_fold = Y_train[train_idx]
        Y_label_fold = Y_label[train_idx]

        train_dataset = list(zip(X_train_fold, X_train2_fold, Y_train_fold, Y_label_fold))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = CAFN().to(device)
        # model = Model.to(device)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=70)

        train_for_val(model, train_loader, X_train_fold, X_train2_fold, Y_train_fold, X_ori, Y_ori, optimizer,
                      num_epochs,train_calculator, scheduler, X_val2=X_ori2, is_train=True, fold=fold)

    train_calculator.transform_format(is_kfold=True)
    ratio = train_calculator.calculate_each_class_absolute_true_rate()
    print(f"[INFO]\t5-fold:each class absolute true rate in train set:{ratio}")

def train_for_test(model, train_loader, X_train, X_train2, targets_train, optimizer, num_epoch, is_train=True):
    for epoch in range(num_epoch):
        model.train()
        loss_Fn=nn.BCEWithLogitsLoss(reduction='mean')
        for batch_inputs, batch_inputs2, batch_targets, batch_y in train_loader:
            loss_fn=loss_Fn
            batch_inputs = batch_inputs.to(device)
            batch_inputs2 = batch_inputs2.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs, batch_inputs2)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train, X_train2)
                targets_train=targets_train.to(device)
                train_loss = loss_Fn(train_outputs, targets_train)
                print(f"[INFO]\tEpoch {epoch + 1}, Train Loss: {train_loss.item()}")
            print(f"[INFO]\tSaving model at epoch {epoch + 1}")
            save_model(model=model, save_path='ckpt', num_epoch=epoch + 1, is_train=is_train)

    train_calculator = Metrics()
    train_predictions = predict_threshold(model, X_train, X_train2, targets_train)
    train_calculator.calculate_metrics(targets_train.cpu(), train_predictions.cpu())
    train_calculator.transform_format()

    train_calculator.accumulate_counts(targets_train.cpu(), train_predictions.cpu())
    ratio = train_calculator.calculate_each_class_absolute_true_rate()
    print(f"[INFO]\teach class absolute true rate in train set:{ratio}")

def model_k_fold_train(X_train, targets_train, X_ori, Y_ori, Y_label, X_train2, X_ori2):
    train_in_k_fold(X_train=X_train, Y_train=targets_train, lr=lr, weight_decay=weight_decay,
                    num_epochs=num_epoch, X_ori=X_ori, Y_ori=Y_ori, Y_label=Y_label, X_train2=X_train2, X_ori2=X_ori2)

def model_test_train(X_train, targets_train, Y_train, X_train2):
    train_dataset = list(zip(X_train, X_train2, targets_train, Y_train))
    train_loader = DataLoader(train_dataset, batch_size=B_S, shuffle=True)
    Model = CAFN()
    model = Model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=test_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=test_weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=test_weight_decay)
    train_for_test(model=model, train_loader=train_loader, X_train=X_train, X_train2=X_train2, targets_train=targets_train,
                   optimizer=optimizer, num_epoch=test_num_epoch, is_train=False)