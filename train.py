import numpy as np
import torch
import sys
import os
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from modelST_CLSToken import MultiHeadEEGModelCLS, deep_cca_loss_nowhiten, deep_cca_loss_svd_debug
from HelpWithMatlab import normalize_sample, filter_classes, riemannian_align_trials
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from load_utils import load_all_subjects
from sklearn.model_selection import train_test_split

from config import (
    in_channels, d_model, num_classes, total_blocks, seg_time,
    batch_size, max_epochs, early_stopping_patience, lr, dropout,
    subject_ids, channels, crop_len, folder, device
)


# # [48 54:58, 61:63] - 9 channels
# in_channels = 9*3
# d_model = 128
# temperature = 0.1
# epochs = 100
# lr = 1e-4
# batch_size = 8

# print(lr)




# subject_ids = range(1, 36)
# channels = [47, 53, 54, 55, 56, 57, 60, 61, 62]
# folder = "C:/Users/vange/Documents/MatlabFiles/ssvep_benchmark_dataset"
# # folder = "C:/Users/vange/Documents/MatlabFiles/BETA_SSVEP/"
# crop_len = 250
# num_classes = 40
# total_blocks = 6
# max_epochs = 50
# early_stopping_patience = 600



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mlflow.set_experiment("SSVEP-LOBO-likeDNN-Bench")

all_subjects = load_all_subjects(folder, subject_ids, crop_len=crop_len, channels=channels)


def run_second_stage(X_train, y_train, X_test, y_test, base_model, sizes, num_classes, sub_idx, seg_time, dropout=0.7, max_epochs=100):


    model =  MultiHeadEEGModelCLS(in_channels=in_channels, d_model=128,H=in_channels,num_layers=4,
                           num_classes=num_classes,time_len=seg_time).to(device)
    model.load_state_dict(base_model.state_dict())

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    best_acc = 0
    patience = 0

    # === TRAINING LOOP ===
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()


            x_hat, _, y_hat = model(xb,xb)

            rec_loss = F.mse_loss(x_hat, xb)
            cls_loss = F.cross_entropy(y_hat, yb)

            loss = 0.01*rec_loss + cls_loss #+ alpha2*cca_loss

            loss.backward()
            optimizer.step()



            total_loss += loss.item() * yb.size(0)
            pred = y_hat.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total

        # === MLflow logging ===
        # mlflow.log_metric(f"finetune_train_loss{sub_idx}", epoch_loss, step=epoch)
        # mlflow.log_metric(f"finetune_train_acc{sub_idx}", epoch_acc, step=epoch)

        
    # === EVALUATION ===
    model.eval()
    with torch.no_grad():
        _,out,out_cls_token = model(X_test.to(device),X_test.to(device))
        pred = torch.argmax(out_cls_token, dim=1).cpu().numpy()
        true = y_test.cpu().numpy()
        acc = (pred == true).mean()
        cm = confusion_matrix(true, pred, labels=np.arange(num_classes))

    return acc, cm




def get_train_test_indices(block, n_targets=40, n_trials=6):
    test_idx = [target * n_trials + block for target in range(n_targets)]
    train_idx = [target * n_trials + trial 
                 for target in range(n_targets) 
                 for trial in range(n_trials) if trial != block]
    return train_idx, test_idx


def train():
    
    for block in range(total_blocks):
        # with mlflow.start_run(run_name=f"0.3s Block_{block+1}"):
        print(f"==== BLOCK {block+1}/{total_blocks} ====")
        acc_matrix = []
        conf_matrix_total = np.zeros((num_classes, num_classes), dtype=int)

        # First-stage global training
        X_all, y_all = [], []
        for subject in subject_ids:
            X, y = all_subjects[subject]
            bs = X.shape[0] // total_blocks
            # idx = list(set(range(X.shape[0])) - set(range(block * bs, (block + 1) * bs)))
            idx = list(range(0, block * bs)) + list(range((block + 1) * bs, X.shape[0]))
            X_all.append(X[idx])
            y_all.append(y[idx])
        X_train_global = torch.cat(X_all)
        print("X_train_global.shape:", X_train_global.shape)
        # X_train_global = X_temp[:,:,:]
        X_train_global = X_train_global.permute(0, 1, 3, 2)     # (B, C, W, T)
        X_train_global = X_train_global.reshape(X_train_global.shape[0], -1, X_train_global.shape[3])  # (B, C*W, T)
        y_train_global = torch.cat(y_all)

        X1 = X_train_global.detach().cpu().numpy()
        # riemannian_align_trials(X_train_global.detach().cpu().numpy())
        # for i in range(X1.shape[0]):
        #     X1[i,:,:] = normalize_sample(X1[i,:,:])
        X2 = torch.from_numpy(X1[:,:,0:seg_time])
        X2 = X2.float()  # ή xb = xb.to(torch.float32)



        model =  MultiHeadEEGModelCLS(in_channels=in_channels, d_model=128,H=in_channels,num_layers=4,
                           num_classes=num_classes,time_len=seg_time).to(device)

        # X_trainG, X_valG, y_trainG, y_valG = train_test_split(
        #     X2, y_train_global, test_size=0.2, random_state=42, stratify=y_train_global
        # )

        train_loader = DataLoader(TensorDataset(X2,y_train_global), batch_size=16, shuffle=True)
        # val_loader = DataLoader(TensorDataset(X_valG, y_valG), batch_size=100, shuffle=True)
        # criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

        best_acc = 0
        patience = 0
        best_val_acc=0
        for epoch in range(100):
            model.train()
            correct, total, epoch_loss = 0, 0, 0
            for xb, yb in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"): #train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                x_hat,_,y_hat= model(xb,xb)

                # loss = criterion(out, yb)
                # loss.backward()
                # optimizer.step()

                rec_loss = F.mse_loss(x_hat, xb)
                cls_loss = F.cross_entropy(y_hat, yb)

                # x_proj, s_proj = model.deep_cca(x_enc, xb_clean)
                # cca_loss = deep_cca_loss_nowhiten(x_proj, s_proj)

                loss = 0.01*rec_loss + cls_loss #+ alpha2*cca_loss

                loss.backward()
                optimizer.step()

                _, preds = torch.max(y_hat, 1)
                correct += (preds == yb).sum().item()
                epoch_loss += loss.item() * yb.size(0)  # sum total loss
                total += yb.size(0)

            acc = correct / total
            avg_loss = epoch_loss / total
            # mlflow.log_metric("global_train_acc", acc, step=epoch)
            # mlflow.log_metric("global_train_loss", avg_loss, step=epoch)

            if acc > best_acc:
                best_acc = acc
                patience = 0
            else:
                patience += 1
            if patience >= early_stopping_patience:
                break


        # Second-stage per subject
        for subject in subject_ids:
            print("Fine-tuning subject", subject)
            X, y = all_subjects[subject]
            bs = X.shape[0] // total_blocks

            # test_idx = list(range(block * bs, (block + 1) * bs))
            # train_idx = list(range(0, block * bs)) + list(range((block + 1) * bs, X.shape[0]))
            train_idx, test_idx = get_train_test_indices(block=block,n_targets=40, n_trials=6)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            X_train = X_train.permute(0, 1, 3, 2)     # (B, C, W, T)
            X_train = X_train.reshape(X_train.shape[0], -1, X_train.shape[3])  # (B, C*W, T)
            X_test = X_test.permute(0, 1, 3, 2)     # (B, C, W, T)
            X_test = X_test.reshape(X_test.shape[0], -1, X_test.shape[3])  # (B, C*W, T)

            acc, cm = run_second_stage(X_train[:,:,0:seg_time], y_train, X_test[:,:,0:seg_time], y_test, model,
                                       (X.shape[1], X.shape[2], 1), num_classes,subject,seg_time)

            # acc, cm = run_second_stage(X_train[:,:,:], y_train, X_test[:,:,:], y_test, model,
            #                            (X.shape[1], X.shape[2], X.shape[3]), num_classes)
            acc_matrix.append(acc)
            # conf_matrix_total += cm
           #mlflow.log_metric(f"acc_subject_{subject:02d}", acc)

        #mlflow.log_metric("block_mean_acc", float(np.mean(acc_matrix)))
        print("block_mean_acc", float(np.mean(acc_matrix)))