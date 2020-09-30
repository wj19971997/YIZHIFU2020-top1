import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from WJ_code.target_encoding import get_target_encoding_tr_ts
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings("ignore")


def load_dataset(DATA_PATH, test_name='testb'):
    # train
    train_label = pd.read_csv(DATA_PATH + '/train/train_label.csv')
    train_base = pd.read_csv(DATA_PATH + '/train/train_base.csv')
    train_op = pd.read_csv(DATA_PATH + '/train/train_op.csv')
    train_trans = pd.read_csv(DATA_PATH + '/train/train_trans.csv')

    # test
    test_base = pd.read_csv(DATA_PATH + '/test/{}_base.csv'.format(test_name))
    test_op = pd.read_csv(DATA_PATH + '/test/{}_op.csv'.format(test_name))
    test_trans = pd.read_csv(DATA_PATH + '/test/{}_trans.csv'.format(test_name))

    return train_label, train_base, test_base, train_op, train_trans, test_op, test_trans


def transform_time(x):
    day = int(x.split(' ')[0])
    hour = int(x.split(' ')[2].split('.')[0].split(':')[0])
    minute = int(x.split(' ')[2].split('.')[0].split(':')[1])
    second = int(x.split(' ')[2].split('.')[0].split(':')[2])
    return 86400 * day + 3600 * hour + 60 * minute + second


def data_preprocess(DATA_PATH, test_name='testb'):
    train_label, train_base, test_base, train_op, train_trans, test_op, test_trans = load_dataset(DATA_PATH=DATA_PATH, test_name=test_name)
    # 拼接数据
    train_df = train_base.copy()
    test_df = test_base.copy()
    train_df = train_label.merge(train_df, on=['user'], how='left')
    del train_base, test_base

    op_df = pd.concat([train_op, test_op], axis=0, ignore_index=True)
    trans_df = pd.concat([train_trans, test_trans], axis=0, ignore_index=True)
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    del train_op, test_op, train_df, test_df
    # 时间维度的处理
    op_df['days_diff'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[0]))
    trans_df['days_diff'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[0]))
    op_df['timestamp'] = op_df['tm_diff'].apply(lambda x: transform_time(x))
    trans_df['timestamp'] = trans_df['tm_diff'].apply(lambda x: transform_time(x))
    op_df['hour'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    trans_df['hour'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    trans_df['week'] = trans_df['days_diff'].apply(lambda x: x % 7)
    # 排序
    trans_df = trans_df.sort_values(by=['user', 'timestamp'])
    op_df = op_df.sort_values(by=['user', 'timestamp'])
    trans_df.reset_index(inplace=True, drop=True)
    op_df.reset_index(inplace=True, drop=True)

    gc.collect()

    return data, op_df, trans_df


def balance_type(data, col):
    data.loc[data[col].notnull(), col] = data.loc[data[col].notnull(), col].apply(
        lambda x: str(x).split(' ')[1]).astype(int)
    data[col] = data[col].fillna(-1).astype(int)

    return data


def base_preprocess(df):
    balance_cols = ['balance', 'balance_avg', 'balance1', 'balance1_avg', 'balance2', 'balance2_avg']
    product_cols = ['product{}_amount'.format(i) for i in range(1, 7)]
    order_cols = balance_cols + product_cols
    for col in order_cols:
        df = balance_type(df, col)

    df['product7_fail_ratio'] = df['product7_fail_cnt'] / df['product7_cnt']

    df['city_count'] = df.groupby(['city'])['user'].transform('count')
    df['province_count'] = df.groupby(['province'])['user'].transform('count')
    df['balance_count'] = df.groupby(['balance'])['user'].transform('count')
    df['ip_cnt_count'] = df.groupby(['ip_cnt'])['user'].transform('count')
    df['using_time_count'] = df.groupby(['using_time'])['user'].transform('count')

    cat_cols = []
    for col in tqdm([f for f in df.select_dtypes('object').columns if f not in ['user'] + order_cols]):
        le = LabelEncoder()
        df[col].fillna('-1', inplace=True)
        df[col] = le.fit_transform(df[col])
        cat_cols.append(col)

    all_features = list(df.columns)
    all_features.remove('user')
    all_features.remove('label')

    return df, all_features


def get_seq(data, cols_1, cols_2=[], seq_len=200):
    data = data.groupby('user').head(seq_len).fillna(0)  # .fillna(-1)
    df = pd.DataFrame()
    df['user'] = data['user'].unique()
    emb_nuniq = []
    for col in cols_1:
        # print(data[['user', col, 'timestamp']])
        nuniq = data[col].nunique() + 1
        match = dict(zip(data[col].unique(), range(1, nuniq)))
        emb_nuniq.append(nuniq)
        data[col] = data[col].map(match).astype(int)
        tmp = data.groupby('user')[col].agg(list).reset_index()
        # print(tmp)
        df = df.merge(tmp, on='user', how='left')

    for col in cols_2:
        data[col] = (data[col] - data[col].mean()) / data[col].std()
        tmp = data.groupby('user')[col].agg(list).reset_index()
        df = df.merge(tmp, on='user', how='left')

    return df, emb_nuniq


def get_target_encoding(base_df, op_df, trans_df, kfold=5):
    base_df['city_level'] = base_df['city'].map(str) + '_' + base_df['level'].map(str)
    base_df['city_balance_avg'] = base_df['city'].map(str) + '_' + base_df['balance_avg'].map(str)

    train, test = base_df[base_df.label.notnull()], base_df[base_df.label.isnull()]

    init_f = list(train.columns)
    card_cnt = ['card_a_cnt', 'card_b_cnt', 'card_c_cnt', 'card_d_cnt']
    target_encode_cols = ['province', 'city', 'city_level', 'city_balance_avg', 'age', ]
    target_encode_cols = target_encode_cols + card_cnt
    # base_df[target_encode_cols].fillna(-1, inplace=True)

    # trans & op 目标编码
    trans_feature = ['platform', 'tunnel_in', 'tunnel_out', 'type1', 'type2', 'amount', ]
    op_features = ['op_type', 'op_device', 'channel', 'op_mode', ]

    agg_lst, not_adp = ['mean'], True
    train, test, trans_df, op_df, trans_new_features, op_new_features = get_target_encoding_tr_ts(train, test, trans_df,
                                                                                                  op_df,
                                                                                                  target_encode_cols,
                                                                                                  trans_feature,
                                                                                                  op_features,
                                                                                                  kfold=kfold,
                                                                                                  not_adp=not_adp,
                                                                                                  agg_lst=agg_lst)

    target_features = list(set(train.columns) - set(init_f))

    new_base = pd.concat([train, test], axis=0)
    for feature in target_features:
        new_base[feature].fillna(new_base[feature].mean(), inplace=True)
    print(new_base[target_features])

    return new_base, target_features, op_df, trans_df, trans_new_features, op_new_features


def get_data(DATA_PATH, op_cols, trans_cols, test_name='testb'):
    base, op, trans = data_preprocess(DATA_PATH, test_name)

    op.rename(columns={'ip': 'op_ip', 'ip_3': 'op_ip_3'}, inplace=True)
    trans.rename(columns={'ip': 'trans_ip', 'ip_3': 'trans_ip_3'}, inplace=True)

    base, target_features, op, trans, trans_new_features, op_new_features = get_target_encoding(base, op, trans,
                                                                                                kfold=5)

    # op file
    op_df, op_emb_nuniq = get_seq(op, op_cols, cols_2=op_new_features, seq_len=200)

    # trans file
    trans_df, trans_emb_nuniq = get_seq(trans, trans_cols, cols_2=trans_new_features, seq_len=100)

    # Normalization of base file
    base, all_features = base_preprocess(base)
    base_df = base[['user', 'label']]
    cate_features = ['sex', 'provider', 'level', 'verified', 'regist_type', 'agreement1', 'agreement2', 'agreement3',
                     'agreement4', 'service3'] + ['province', 'city', ]
    oh_enc = OneHotEncoder()
    df_cate = oh_enc.fit_transform(base[cate_features]).toarray()
    enc_features = ['oh_enc{}'.format(i) for i in range(df_cate.shape[1])]
    df_cate = pd.DataFrame(df_cate, columns=enc_features)
    base_df = pd.concat([base_df, df_cate], axis=1)
    del df_cate
    gc.collect()

    numerical_features = list(set(all_features) - set(cate_features))
    base_df = pd.concat([base_df, base[numerical_features]], axis=1)

    # Normalization
    for feature in enc_features + numerical_features + target_features:
        base_df[feature] = (base_df[feature] - base_df[feature].mean()) / base_df[feature].std()

    return base_df, (op_df, op_emb_nuniq), (trans_df, trans_emb_nuniq)


def one_hot(labels, num_classes):
    labels = np.squeeze(labels).astype(int)
    if labels.ndim == 0:
        arr = np.zeros(num_classes)
        arr[labels] = 1
        return arr
    batch_size = labels.shape[0]
    idxs = np.arange(0, batch_size, 1)
    arr = np.zeros([batch_size, num_classes])
    arr[idxs, labels] = 1
    return arr


class TYDataSet(object):
    def __init__(self, data, with_label=True, y=None, n_class=2,
                 op_cols=['op_type', 'op_mode', 'op_device', 'channel'],
                 trans_cols=['tunnel_in', 'tunnel_out', 'type1', 'type2', 'platform']):
        self.base = torch.from_numpy(data.select_dtypes(float).values)
        self.op, self.op_seq_len = self.pad_seq(data[op_cols], op_cols)
        self.trans, self.trans_seq_len = self.pad_seq(data[trans_cols], trans_cols)
        self.n_class = n_class

        self.with_label = with_label
        if self.with_label:
            self.y = torch.from_numpy(y.values).squeeze()
            # self.y = y.values

    def pad_seq(self, data, cols):
        data.fillna(-1, inplace=True)
        ts = None
        seq_len = None
        for col in cols:
            tmp = data[col].values.tolist()
            # tmp = list(map(lambda x: torch.Tensor(x) if x != -1 else torch.Tensor([]), tmp))
            tmp = list(map(lambda x: torch.Tensor(x) if x != -1 else torch.Tensor([0]), tmp))
            if seq_len is None:
                seq_len = list(map(len, tmp))

            tmp = torch.nn.utils.rnn.pad_sequence(tmp)
            tmp = tmp.view(tmp.size(0), tmp.size(1), 1).permute(1, 0, 2)
            # print(col, tmp.shape)
            if ts is None:
                ts = tmp
            else:
                ts = torch.cat([ts, tmp], dim=2)
        # print(ts.shape)
        return ts, seq_len

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        tup = (self.base[item], self.op[item], self.trans[item])
        if self.with_label:
            return tup, self.y[item], self.op_seq_len[item], self.trans_seq_len[item]
        else:
            return tup, self.op_seq_len[item], self.trans_seq_len[item]


def get_DataLoader(data, op_cols, trans_cols, y=None, batch_size=512, shuffle=False, with_label=True):
    data_set = TYDataSet(data, y=y, with_label=with_label, op_cols=op_cols, trans_cols=trans_cols)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    return data_loader