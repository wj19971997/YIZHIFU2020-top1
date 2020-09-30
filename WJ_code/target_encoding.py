import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import gc
from time import time
import warnings

warnings.filterwarnings("ignore")

def target_encoding(file, train, test, feats, k, not_adp=True, agg_lst=['mean']):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=1997)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    tt_file = train[['user', 'label']].merge(file, on='user', how='left')

    te_features = []
    file_new_features = []
    for feat in tqdm(feats):
        col_name = feat + '_te'
        file_new_features.append(col_name)
        # te_features.append(col_name)
        file[col_name] = 0
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
            tmp_users = train.iloc[trn_idx]['user'].values
            tmp_file = file[file.user.isin(tmp_users)]
            tmp_file = tmp_file.merge(train[['user', 'label']], on='user', how='left')

            if not_adp:
                match = tmp_file.groupby(feat)['label'].mean()
            else:
                match = tmp_file.groupby([feat, 'user'])['label'].agg(eu_sum='sum', eu_count='count').reset_index()
                match['eu_mean'] = match['eu_sum'] / match['eu_count']
                match = match['eu_mean'].groupby(match[feat]).mean()

            # match = tmp_file.groupby([feat, 'user'])['label'].agg(eu_sum='sum', eu_count='count').reset_index()
            # match['eu_mean'] = match['eu_sum'] / match['eu_count']
            # match = match['eu_mean'].groupby(match[feat]).mean()

            tmp_users = train.iloc[val_idx]['user'].values
            tmp_file = file[file.user.isin(tmp_users)]
            file.loc[file.user.isin(tmp_users), col_name] = file.loc[file.user.isin(tmp_users), feat].map(match)
            tmp_file[col_name] = tmp_file[feat].map(match)
            for agg_ in agg_lst:
                tmp = tmp_file.groupby('user')[col_name].agg(agg_)
                train.loc[train.fold == fold_, col_name + '_' + agg_] = train.loc[train.fold == fold_, 'user'].map(tmp)
            # tmp = tmp_file.groupby('user')[col_name].mean()
            # train.loc[train.fold == fold_, col_name] = train.loc[train.fold == fold_, 'user'].map(tmp)

            # fillna
            # global_mean = tmp.mean()
            # train.loc[train.fold == fold_, col_name] = train.loc[train.fold == fold_, col_name].fillna(global_mean)

        if not_adp:
            match = tt_file.groupby(feat)['label'].mean()
        else:
            match = tt_file.groupby([feat, 'user'])['label'].agg(eu_sum='sum', eu_count='count').reset_index()
            match['eu_mean'] = match['eu_sum'] / match['eu_count']
            match = match['eu_mean'].groupby(match[feat]).mean()

        # match = tt_file.groupby([feat, 'user'])['label'].agg(eu_sum='sum', eu_count='count').reset_index()
        # match['eu_mean'] = match['eu_sum'] / match['eu_count']
        # match = match['eu_mean'].groupby(match[feat]).mean()

        tmp_file = file[file.user.isin(test['user'].values)]
        file.loc[file.user.isin(test['user'].values), col_name] = file.loc[
            file.user.isin(test['user'].values), feat].map(match)
        tmp_file[col_name] = tmp_file[feat].map(match)
        for agg_ in agg_lst:
            tmp = tmp_file.groupby('user')[col_name].agg(agg_)
            test[col_name + '_' + agg_] = test['user'].map(tmp)
            te_features.append(col_name + '_' + agg_)

    del train['fold']
    gc.collect()

    return train, test, file, file_new_features


def kfold_stats_feature(train, test, feats, k):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=1997)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in tqdm(feats):
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                # global_mean = train[f].mean()
                # train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            fill_mean = train[colname].mean()
            train[colname] = train[colname].fillna(fill_mean)

            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            # global_mean = train[f].mean()
            # test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].fillna(fill_mean)
            test[colname] = test[colname].astype(float)

    del train['fold']

    return train, test


def get_target_encoding_tr_ts(train, test, trans_df, op_df,
                              base_feature, trans_feature, op_feature, kfold, not_adp=True, agg_lst=['mean']):
    train, test = kfold_stats_feature(train, test, base_feature, kfold)
    train.drop(['city_level', 'city_balance_avg', ], axis=1, inplace=True)
    test.drop(['city_level', 'city_balance_avg', ], axis=1, inplace=True)
    train, test, trans_df, trans_new_features = target_encoding(trans_df, train, test, trans_feature, kfold,
                                                                not_adp=not_adp, agg_lst=agg_lst)
    train, test, op_df, op_new_features = target_encoding(op_df, train, test, op_feature, kfold, not_adp=not_adp,
                                                          agg_lst=agg_lst)

    return train, test, trans_df, op_df, trans_new_features, op_new_features