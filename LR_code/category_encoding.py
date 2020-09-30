#/home/lirui/anaconda/envs python3.6.5
# -*- coding: UTF-8 -*-
"""
@Author: LiRui
@Date: 2020-09-03 16:19:01
@LastEditTime: 2020-09-04 19:38:51
@Description: 类别特征编码, 包括目标编码, label_encode, 以及未使用的WOE, MEE编码等
@File: /YIZHIFU_2020_Cloud/LR_code/category_encoding.py

"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold ,KFold
from tqdm import tqdm
import gc
warnings.filterwarnings('ignore')

def target_encoding(file, train, test, feats, k, prefix, not_adp=True, agg_lst=['mean'], seed=2020):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    tt_file = train[['user', 'label']].merge(file, on='user', how='left')

    te_features = []
    for feat in tqdm(feats, desc='Target encoding for {} feature '.format(prefix)):
        col_name = feat + '_te'
        # te_features.append(col_name)
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

            tmp_users = train.iloc[val_idx]['user'].values
            tmp_file = file[file.user.isin(tmp_users)]
            tmp_file[col_name] = tmp_file[feat].map(match)
            for agg_ in agg_lst:
                tmp = tmp_file.groupby('user')[col_name].agg(agg_)
                train.loc[train.fold == fold_, col_name + '_' + agg_] = train.loc[train.fold == fold_, 'user'].map(tmp)

        if not_adp:
            match = tt_file.groupby(feat)['label'].mean()
        else:
            match = tt_file.groupby([feat, 'user'])['label'].agg(eu_sum='sum', eu_count='count').reset_index()
            match['eu_mean'] = match['eu_sum'] / match['eu_count']
            match = match['eu_mean'].groupby(match[feat]).mean()

        tmp_file = file[file.user.isin(test['user'].values)]
        tmp_file[col_name] = tmp_file[feat].map(match)
        for agg_ in agg_lst:
            tmp = tmp_file.groupby('user')[col_name].agg(agg_)
            test[col_name + '_' + agg_] = test['user'].map(tmp)


    del train['fold']
    gc.collect()
    # print(train[te_features])

    return train, test

def dis(data, k=10):
    min_, max_ = data.min(), data.max()
    interval = (max_ - min_) / k
    for i in range(k):
        data.loc[(data >= min_ + i * interval) & (data <= min_ + (i + 1) * interval)] = i

    return data

def CalWOE(df, fea, label):
    eps = 0.000001
    gbi = pd.crosstab(df[fea], df[label]) + eps
    gb = df[label].value_counts() + eps
    gbri = gbi / gb
    gbri['woe'] = np.log(gbri[1] / gbri[0])

    return gbri['woe']

def kfold_stats_feature(train, test, feats, k, seed):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in tqdm(feats, desc='Target encoding for base feature'):
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
                global_mean = tmp_trn[f].mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = train[f].mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test


#woe编码, 暂时未加入做特征
def woe_feature(train, test, feats, k):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_woe'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                #order_label = tmp_trn.groupby([feat])[f].mean()
                order_label = CalWOE(tmp_trn, feat, f)
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = order_label.mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_woe'
            test[colname] = None
            #order_label = train.groupby([feat])[f].mean()
            order_label = CalWOE(train, feat, f)
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = order_label.mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test

def label_encode(df, order_cols):
    # LabelEncoder
    cat_cols = [f for f in df.select_dtypes('object').columns if f not in ['user'] + order_cols]
    for col in cat_cols:
        le = LabelEncoder()
        df[col].fillna('-1', inplace=True)
        df[col] = le.fit_transform(df[col])
        #cat_cols.append(col)
    return  df

#catboost 类别编码,暂时未加入做特征
def cat_encoding(train, test, k ,feature):
    #feature = [f for f in train.select_dtypes('object').columns if f not in ['user']]
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    enc = CatEncode.CatBoostEncoder()

    for feat in feature:
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_cat_enc'

            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                #order_label = tmp_trn.groupby([feat])[f].mean()
                order_label = enc.fit_transform(tmp_trn[feat], tmp_trn['label']).values.squeeze()
                enc_dic = dict(zip(tmp_trn[feat], order_label))
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(enc_dic)
                # fillna
                global_mean = order_label.mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_cat_enc'
            test[colname] = None
            order_label = enc.fit_transform(train[feat], train['label']).values.squeeze()
            enc_dic = dict(zip(train[feat], order_label))
            test[colname] = test[feat].map(enc_dic)
            # fillna
            global_mean = order_label.mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
        #del train[feat]
    del train['fold']
    return train, test

def order_encode(df, col):
    """

    :param df:      Dataframe
    :param col:     feature
    :description:   对有序类别变量顺序编码
    :return:
    """
    df.loc[df[col].notnull(), col] = df.loc[df[col].notnull(), col].apply(lambda x: str(x).split(' ')[1]).astype(int)
    df[col] = df[col].fillna(-1).astype(int)

    return df

#多种类别编码for base feature, 暂时未加入做特征
def cat_encode_list(train, test, k, feature):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    encoder_lst = [ WOEEncoder(), MEstimateEncoder(), JamesSteinEncoder(), LeaveOneOutEncoder(),
                   CatBoostEncoder()]
    enc_name = ['woe_enc', 'me_enc', 'js_enc', 'loo_enc', 'cbt_enc']
    for feat in feature:
        for i, encoder in enumerate(encoder_lst):
            #print(encoder)

            nums_columns = ['label']
            for f in nums_columns:
                colname = feat + '_' + f + '_' + enc_name[i]

                train[colname] = None
                for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                    tmp_trn = train.iloc[trn_idx]
                    # order_label = tmp_trn.groupby([feat])[f].mean()
                    order_label = encoder.fit_transform(tmp_trn[feat], tmp_trn['label'])
                    #enc_dic = dict(zip(tmp_trn[feat], order_label))
                    tmp = train.loc[train.fold == fold_, [feat]]
                    train.loc[train.fold == fold_, colname] = encoder.transform(tmp).values
                    # fillna
                    global_mean = order_label.mean()
                    train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(
                        global_mean)
                train[colname] = train[colname].astype(float)

            for f in nums_columns:
                colname = feat + '_' + f + '_' + enc_name[i]
                test[colname] = None
                order_label = encoder.fit_transform(train[feat], train['label'])

                test[colname] = encoder.transform(test[feat]).values
                # fillna
                global_mean = order_label.mean()
                test[colname] = test[colname].fillna(global_mean)
                test[colname] = test[colname].astype(float)
        encoder_lst = [WOEEncoder(), MEstimateEncoder(), JamesSteinEncoder(), LeaveOneOutEncoder(),
                       CatBoostEncoder()]
    del train['fold']

    return train, test

#多种类别编码for op/trans 表, 暂时未加入做特征
def cat_encode_lst_trans_op(file, train, test, feature, k,):
    print('start category encoding for trans & op feature')
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    tt_file = train[['user', 'label']].merge(file, on='user', how='left')
    encoder_lst = [WOEEncoder(), MEstimateEncoder(), JamesSteinEncoder(), LeaveOneOutEncoder(),
                   CatBoostEncoder()]
    enc_name = [ 'woe_enc', 'me_enc', 'js_enc', 'loo_enc', 'cbt_enc']
    for feat in feature:
        for i, encoder in enumerate(encoder_lst):
            col_name = feat  + '_' + enc_name[i]
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_users = train.iloc[trn_idx]['user'].values
                tmp_file = file[file.user.isin(tmp_users)]
                tmp_file = tmp_file.merge(train[['user', 'label']], on='user', how='left')

                #match = tmp_file.groupby(feat)['label'].mean()
                match = encoder.fit_transform(tmp_file[feat], tmp_file['label'])

                tmp_users = train.iloc[val_idx]['user'].values
                tmp_file = file[file.user.isin(tmp_users)]
                #tmp_file[col_name] = tmp_file[feat].map(match)
                tmp_file[col_name] = encoder.transform(tmp_file[feat]).values
                tmp = tmp_file.groupby('user')[col_name].mean()
                train.loc[train.fold == fold_, col_name] = train.loc[train.fold == fold_, 'user'].map(tmp)

            #match = tt_file.groupby(feat)['label'].mean()
            match = encoder.fit_transform(tt_file[feat], tt_file['label'])
            tmp_file = file[file.user.isin(test['user'].values)]
            #tmp_file[col_name] = tmp_file[feat].map(match)
            tmp_file[col_name] = encoder.transform(tmp_file[feat]).values
            tmp = tmp_file.groupby('user')[col_name].mean()
            test[col_name] = test['user'].map(tmp)

        encoder_lst = [WOEEncoder(), MEstimateEncoder(), JamesSteinEncoder(), LeaveOneOutEncoder(),
                       CatBoostEncoder()]
    del train['fold']

    return train, test




def get_target_encoding_tr_ts(train, test, trans_df, op_df, trans_op_df, woe_fea_lst,
                              base_feature, trans_feature, op_feature, trans_op_feature, folds, seed):

    train, test = kfold_stats_feature(train, test, base_feature, folds, seed)
    #train, test = cat_encode_list(train, test, k=folds, feature=base_feature,)

    #train, test = woe_feature(train, test, base_feature, folds)
    drop_feature = ['city_level', 'city_balance_avg',]
    #cat_feature = ['city', 'sex', 'verified', 'provider', 'regist_type',]
    agg_lst = ['mean', ]#'std', 'median', 'max', 'min']

    train, test = target_encoding(trans_df, train, test, trans_feature, folds, prefix='trans', not_adp=True, agg_lst=agg_lst, seed=seed)
    agg_lst = ['mean',  'std', 'max', 'min', 'sum', 'median', ]
    train, test = target_encoding(op_df, train, test, op_feature, folds, prefix='op', not_adp=True, agg_lst=agg_lst, seed=seed)
    del trans_op_df['label']
    train, test = target_encoding(trans_op_df, train, test, trans_op_feature, folds, prefix='trans_op', not_adp=True, agg_lst=agg_lst, seed=seed)
    #train, test = cat_encoding(train, test, folds, cat_feature)
    #有大小关系类别特征单独编码
    order_cols = ['balance', 'balance_avg', 'balance1', 'balance1_avg', 'balance2', 'balance2_avg'] + \
                    ['product{}_amount'.format(i) for i in range(1, 7)]
    for col in order_cols:
        train, test = order_encode(train, col), order_encode(test, col)
    #类别特征硬编码
    train, test = label_encode(train, order_cols), label_encode(test, order_cols)
    #去除丢掉的特征
    train.drop(drop_feature, axis=1, inplace=True)
    test.drop(drop_feature, axis=1, inplace=True)
    print('Category feature encoding finished.........')

    return train, test

