import os
import pandas as pd
import numpy as np
import random
from WJ_code.seed import seed
from WJ_code.Model import Model
from WJ_code.get_dataloader import get_DataLoader, get_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
import gc
from tqdm import tqdm
import copy
from time import time
import warnings

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings("ignore")

seed(1997)

def get_result(folds=5, bs=32, epoch_num=50, lr=0.01, data_path = './data', sub_name='sub.csv', test_name='testb'):
    op_cols = ['op_type', 'op_mode', 'op_device', 'channel', ]
    trans_cols = ['tunnel_in', 'tunnel_out', 'type1', 'type2', 'platform'] + ['amount']
    op_emb_size_lst = [3, 3, 3, 3]
    trans_emb_size_lst = [1, 1, 1, 1, 1, 3]
    op_te_cols = ['{}_te'.format(col) for col in op_cols]
    trans_te_cols = ['{}_te'.format(col) for col in trans_cols]

    base, (op, op_emb_nuniq), (trans, trans_emb_nuniq) = get_data(data_path, op_cols, trans_cols, test_name=test_name)
    base_input_dim = len(base.columns) - 2
    data = pd.merge(base, op, on='user', how='left')
    data = data.merge(trans, on='user', how='left')

    train, test = data[data.label.notnull()], data[data.label.isnull()]
    train_y = train[['label']]

    sub = test[['user']].copy()
    train.drop(columns=['user', 'label'], inplace=True)
    test.drop(columns=['user', 'label'], inplace=True)

    prob_t, prob_v = np.zeros(len(test)), np.zeros(len(train))
    roc_lst = []

    op_cols += op_te_cols
    trans_cols += trans_te_cols
    test_loader = get_DataLoader(test, with_label=False, op_cols=op_cols, trans_cols=trans_cols,
                                 shuffle=False, batch_size=1024
                                 )
    skf = StratifiedKFold(n_splits=folds, random_state=1997, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train_y)):
        print('The fold ', fold)
        train_X, y = train.iloc[train_idx], train_y.iloc[train_idx]
        val_X, val_y = train.iloc[val_idx], train_y.iloc[val_idx]

        train_loader = get_DataLoader(train_X, y=y, with_label=True, op_cols=op_cols, trans_cols=trans_cols,
                                      batch_size=bs, shuffle=True
                                      )
        val_loader = get_DataLoader(val_X, y=val_y, with_label=True, op_cols=op_cols, trans_cols=trans_cols,
                                    batch_size=4096, # len(val_y),
                                    shuffle=False
                                    )

        model = Model(base_input_dim, op_emb_nuniq, trans_emb_nuniq,
                      op_emb_size_lst=op_emb_size_lst, trans_emb_size_lst=trans_emb_size_lst,
                      )

        model.to(device=device)
        model = torch.nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        total_step = len(train_loader)
        best_roc = 0
        for epoch in range(epoch_num):
            for i, ((base, op, trans), labels, op_sl, trans_sl) in enumerate(train_loader):
                model.train()
                labels = labels.long().to(device)
                outputs = model(base.to(device), op.to(device), trans.to(device), op_sl, trans_sl)

                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    outputs = None
                    for ((base, op, trans), labels_batch, op_sl, trans_sl) in val_loader:
                        if outputs is None:
                            outputs = F.softmax(model(base.to(device), op.to(device), trans.to(device), op_sl,
                                                      trans_sl)).cpu().detach().numpy()
                            labels = labels_batch.detach().numpy()
                        else:
                            outputs = np.concatenate([outputs, F.softmax(
                                model(base.to(device), op.to(device), trans.to(device), op_sl,
                                      trans_sl)).cpu().detach().numpy()])
                            labels = np.concatenate([labels, labels_batch.detach().numpy()])
                    preds = outputs[:, 1]  # .argmax(axis=1)
                    roc = roc_auc_score(labels, preds)

                    if roc > best_roc:
                        best_roc = roc
                        best_model = copy.deepcopy(model)
                if i % (total_step - 1) == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, val_roc: {}, tmp_best_val_roc: {}'
                          .format(epoch + 1, epoch_num, i + 1, total_step, loss.item(), roc, best_roc))


        with torch.no_grad():
            model.eval()
            outputs = None
            for ((base, op, trans), labels_batch, op_sl, trans_sl) in val_loader:
                if outputs is None:
                    outputs = F.softmax(
                        best_model(base.to(device), op.to(device), trans.to(device), op_sl,
                                   trans_sl)).cpu().detach().numpy()
                    labels = labels_batch.detach().numpy()
                else:
                    outputs = np.concatenate([outputs, F.softmax(
                        best_model(base.to(device), op.to(device), trans.to(device), op_sl,
                                   trans_sl)).cpu().detach().numpy()])
                    labels = np.concatenate([labels, labels_batch.detach().numpy()])

            val_preds = outputs[:, 1]
            roc = roc_auc_score(labels, val_preds)
            print('\n[+] The fold {} roc: {}'.format(fold, roc))

        outputs = None
        for (base, op, trans), op_sl, trans_sl in test_loader:
            if outputs is None:
                outputs = F.softmax(best_model(base.to(device), op.to(device), trans.to(device), op_sl,
                                               trans_sl)).cpu().detach().numpy()
            else:
                outputs = np.concatenate((outputs, F.softmax(
                    best_model(base.to(device), op.to(device), trans.to(device), op_sl,
                               trans_sl)).cpu().detach().numpy()))
        test_preds = outputs[:, 1]

        model_save_path = './nn_models/'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(best_model, model_save_path + 'model_fold{}.pkl'.format(fold))

        prob_t += test_preds / folds
        prob_v[val_idx] += val_preds
        roc_lst.append(roc)

    print('[+] The roc lst: ', roc_lst)

    roc = roc_auc_score(train_y, prob_v)
    print('[+] The roc of all train: ', roc)

    sub['prob'] = prob_t
    prob_save_path = './nn_prob/'
    if not os.path.exists(prob_save_path):
        os.makedirs(prob_save_path)

    sub.to_csv(prob_save_path + sub_name, index=False)
    print(sub)

    np.save(prob_save_path + 'val.npy', prob_v)

