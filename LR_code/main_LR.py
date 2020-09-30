# coding: utf-8
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold ,KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import gc
from LR_code.category_encoding import *
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
#from LR_code.lgb_model import *
from LR_code.xgb_model import *
from LR_code.utils import *
from LR_code.vector_feature import *
from LR_code.feature_engineering import *
#from fea_engineering_adp import *
#from LR_code.cbt_models import *
#from  gen_trans_fea_separation_month import *
import os
import argparse
# import category_encoders as CatEncoder


def load_dataset(DATA_PATH):

    train_label = pd.read_csv(DATA_PATH + '/train/train_label.csv')
    train_base = pd.read_csv(DATA_PATH + '/train/train_base.csv')
    test_base = pd.read_csv(DATA_PATH + '/test/testb_base.csv')

    train_op = pd.read_csv(DATA_PATH + '/train/train_op.csv')
    train_trans = pd.read_csv(DATA_PATH + '/train/train_trans.csv')
    test_op = pd.read_csv(DATA_PATH + '/test/testb_op.csv')
    test_trans = pd.read_csv(DATA_PATH + '/test/testb_trans.csv')

    return train_label, train_base, test_base, train_op, train_trans, test_op, test_trans

def transform_time(x):
    day = int(x.split(' ')[0])
    hour = int(x.split(' ')[2].split('.')[0].split(':')[0])
    minute = int(x.split(' ')[2].split('.')[0].split(':')[1])
    second = int(x.split(' ')[2].split('.')[0].split(':')[2])
    return 86400*day+3600*hour+60*minute+second

def data_preprocess(DATA_PATH):
    train_label, train_base, test_base, train_op, train_trans, test_op, test_trans = load_dataset(DATA_PATH=DATA_PATH)
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
    #op_df['hour'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    op_df['week'] = op_df['days_diff'].apply(lambda x: x % 7)
    # 排序
    trans_df = trans_df.sort_values(by=['user', 'timestamp'])
    op_df = op_df.sort_values(by=['user', 'timestamp'])
    trans_df.reset_index(inplace=True, drop=True)
    op_df.reset_index(inplace=True, drop=True)

    gc.collect()
    return data, op_df, trans_df








def main_lr(window, num, seed, trans_behavior_fea, args,):
    seed_everything(args.SEED)


    DATA_PATH = './data'
    print('Loading data......')
    train_base = pd.read_csv(DATA_PATH + '/train/train_base.csv')
    train_label = pd.read_csv(DATA_PATH + '/train/train_label.csv')
    data, op_df, trans_df = data_preprocess(DATA_PATH=DATA_PATH)
    # merge_trans_op
    trans_, op_ = gen_trans_op_label(trans_df, op_df, train_label)
    trans_op_df = trans_op_merge(trans_, op_)
    trans_op_df = gen_fea_trans_op_df(trans_op_df)
    #op表和trans表提取组合特征
    op_df = gen_fea_op_df(op_df,)
    trans_df = gen_fea_trans_df(trans_df)

    print('Start feature engineering......')
    data, cross_feature = gen_features(data, op_df, trans_df, trans_op_df,  window, num)
    data['city_level'] = data['city'].map(str) + '_' + data['level'].map(str)
    data['city_balance_avg'] = data['city'].map(str) + '_' + data['balance_avg'].map(str)
    data['has_trans'] = pd.factorize(data['has_trans'])[0]
    # 类别编码
    print('Start category encoding......')
    train_data = data[~data['label'].isnull()].copy()
    target = train_data['label']
    test_data = data[data['label'].isnull()].copy()
    # 目标编码特征，base & trans & op
    card_cnt = ['card_a_cnt', 'card_b_cnt', 'card_c_cnt', 'card_d_cnt']
    target_encode_cols = ['province', 'city', 'city_level', 'city_balance_avg', 'age',
                          'service1_amt', 'ip_cnt', 'login_days_cnt', 'login_cnt_avg', 'acc_count',
                          ]
    target_encode_cols = target_encode_cols + card_cnt
    # trans & op 目标编码
    trans_feature = ['platform', 'tunnel_in', 'tunnel_out', 'type1', 'type2', 'amount',
                     'platform_amount', 'type_amount', 'tunnel_io_amount',
                     'type1_amount', 'type2_amount', 'tunnel_in_amount', 'tunnel_out_amount'
                     ]
    op_features = ['op_type', 'op_device', 'channel', 'op_mode',
                   'op_ip',
                   'net_type',
                   'op_ip_3',
                   ]
    woe_fea_lst = [f for f in train_base.select_dtypes('object').columns if f not in ['user']]

    trans_op_features = ['week_property', 'hour_property', 'day_property', 'day_week_property',
                         'day_hour_property','days_diff', 'week', 'hour']
    # 训练集和测试集目标编码
    train_data, test_data = get_target_encoding_tr_ts(train_data, test_data, trans_df, op_df, trans_op_df,  woe_fea_lst,
                                                      target_encode_cols, trans_feature, op_features, trans_op_features,
                                                      folds=5, seed=seed)


    train_data, test_data = province_binary(train_data), province_binary(test_data)
    train_data, test_data = train_data.fillna(-999, ), test_data.fillna(-999, )
    train_data['trans_ratio'] = train_data['trans_count'] / train_data['op_count']
    test_data['trans_ratio'] = test_data['trans_count'] / test_data['op_count']
    # train_data.to_hdf('../data/train_data.hdf', 'w', complib='blosc', complevel=5)
    # test_data.to_hdf('../data/test_data.hdf', 'w', complib='blosc', complevel=5)
    user_behavior_fea = ['user_trans_trans_ip_3_null_cnt', 'user_trans_trans_ip_3_null_ratio',
        'has_trans', 'last_beahvior', 'first_beahvior', 'last_days_diff_trans', 'first_days_diff_trans',
          'last_hour_trans', 'first_hour_trans', 'last_week_trans', 'first_week_trans', 'last_time_trans',
           'first_time_trans','trans_count', 'op_count','trans_ratio']


    
    

    if not os.path.exists('./xgb_seed{}/'.format(seed)):
        os.makedirs('./xgb_seed{}/'.format(seed))
    # if not os.path.exists('./xgb_fea914/'):
    #     os.makedirs('./xgb_fea914/')
    # if trans_behavior_fea and drop_crossfea == False:
    #     np.save('./xgb_seed{}/xgb_val_1690fea.npy'.format(seed), xgb_oof)
    #     np.save('./xgb_seed{}/xgb_ts_1690fea.npy'.format(seed), xgb_preds)
    # elif not trans_behavior_fea and drop_crossfea == False:
    #     np.save('./xgb_seed{}/xgb_val_1674fea.npy'.format(seed), xgb_oof)
    #     np.save('./xgb_seed{}/xgb_ts_1674fea.npy'.format(seed), xgb_preds)
    # elif not trans_behavior_fea and drop_crossfea == True:
    #     np.save('./xgb_fea924/xgb_val_seed{}.npy'.format(seed), xgb_oof)
    #     np.save('./xgb_fea924/xgb_ts_seed{}.npy'.format(seed), xgb_preds)

    
    if trans_behavior_fea:
        #1690fea
        xgb_preds, xgb_oof, xgb_score = xgb_model(train=train_data, target=target, test=test_data, k=5, lr=0.02, seed=seed)
        np.save('./xgb_seed{}/xgb_val_1690fea.npy'.format(seed), xgb_oof)
        np.save('./xgb_seed{}/xgb_ts_1690fea.npy'.format(seed), xgb_preds)
        #1674fea(去掉用户行为特征)
        xgb_preds, xgb_oof, xgb_score = xgb_model(train=train_data[[col for col in train_data.columns if col not in user_behavior_fea]], 
                                                  target=target, test=test_data[[col for col in test_data.columns if col not in user_behavior_fea]],
                                                  k=5, lr=0.02, seed=seed)
        np.save('./xgb_seed{}/xgb_val_1674fea.npy'.format(seed), xgb_oof)
        np.save('./xgb_seed{}/xgb_ts_1674fea.npy'.format(seed), xgb_preds)
        #914fea
        xgb_preds, xgb_oof, xgb_score = xgb_model(train=train_data[[col for col in train_data.columns if col not in user_behavior_fea and col not in cross_feature]], 
                                                  target=target, test=test_data[[col for col in test_data.columns if col not in user_behavior_fea and col not in cross_feature]],
                                                  k=5, lr=0.02, seed=seed)
        np.save('./xgb_seed{}/xgb_val_914fea.npy'.format(seed), xgb_oof)
        np.save('./xgb_seed{}/xgb_ts_914fea.npy'.format(seed), xgb_preds)
    else:
        #1674fea(去掉用户行为特征)
        xgb_preds, xgb_oof, xgb_score = xgb_model(train=train_data[[col for col in train_data.columns if col not in user_behavior_fea]], 
                                                  target=target, test=test_data[[col for col in test_data.columns if col not in user_behavior_fea]],
                                                  k=5, lr=0.02, seed=seed)
        np.save('./xgb_seed{}/xgb_val_1674fea.npy'.format(seed), xgb_oof)
        np.save('./xgb_seed{}/xgb_ts_1674fea.npy'.format(seed), xgb_preds)
        #914fea
        xgb_preds, xgb_oof, xgb_score = xgb_model(train=train_data[[col for col in train_data.columns if col not in user_behavior_fea and col not in cross_feature]], 
                                                  target=target, test=test_data[[col for col in test_data.columns if col not in user_behavior_fea and col not in cross_feature]],
                                                  k=5, lr=0.02, seed=seed)
        np.save('./xgb_seed{}/xgb_val_914fea.npy'.format(seed), xgb_oof)
        np.save('./xgb_seed{}/xgb_ts_914fea.npy'.format(seed), xgb_preds)



    sub_df = test_data[['user']].copy()
    sub_df['prob'] = xgb_preds
    print(sub_df.head())
    

def get_para():
    parser = argparse.ArgumentParser(description='help info')
    parser.add_argument('--FOLDS', default=5, type=int)
    parser.add_argument('--tfdif_size', default=10, type=int)
    parser.add_argument('--countvec_size', default=10, type=int)
    parser.add_argument('--num', default=1, type=int)
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--SEED', default=1116, type=int)
    args = parser.parse_args()
    print('seed: ', args.seed)
    print('SEED: ', args.SEED)
    print('num: ', args.num)

    return args

if __name__ == '__main__':


    args = get_para()
    main_lr(window=[15, 23, 27], num=args.num, seed=args.seed, trans_behavior_fea=False, args=args, )









