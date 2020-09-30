#/home/lirui/anaconda/envs python3.6.5
# -*- coding: UTF-8 -*-
"""
@Author: LiRui
@Date: 2020/8/10 下午8:13
@LastEditTime: 2020-07-06 23:15:51
@Description: 
@File: feature_engineering.py

"""
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from LR_code.gen_base_fea import *
from gensim.models import Word2Vec
import seaborn as sns
import matplotlib.pyplot as plt
import os
from LR_code.gen_trans_fea import *
from LR_code.trans_op_merge import *
from LR_code.vector_feature import *

def gen_user_amount_features(df):
    group_df = df.groupby(['user'])['amount'].agg({
        'user_amount_mean': 'mean',
       # 'user_amount_std': 'std',
        'user_amount_max': 'max',
        'user_amount_min': 'min',
        'user_amount_sum': 'sum',
        'user_amount_med': 'median',
        'user_amount_cnt': 'count',
        # 'user_amount_q1': lambda x: x.quantile(0.25),
        # 'user_amount_q3': lambda x: x.quantile(0.75),
        #'user_amount_qsub': lambda x: x.quantile(0.75) - x.quantile(0.25)
        #'user_amount_skew': 'skew',
        }).reset_index()
    return group_df

def gen_user_stastic_features(df, col, prefix):
    group_df = df.groupby(['user'])[col].agg({
        '{}_user_{}_mean'.format(prefix, col): 'mean',
       # 'user_amount_std': 'std',
        '{}_user_{}_max'.format(prefix, col): 'max',
        '{}_user_{}_min'.format(prefix, col): 'min',
        '{}_user_{}_sum'.format(prefix, col): 'sum',
        '{}_user_{}_med'.format(prefix, col): 'median',
        '{}_user_{}_cnt'.format(prefix, col): 'count',
        # '{}_user_{}_q1'.format(prefix, col): lambda x: x.quantile(0.25),
        # '{}_user_{}_q3'.format(prefix, col): lambda x: x.quantile(0.75),
        # '{}_user_{}_qsub'.format(prefix, col): lambda x: x.quantile(0.75) - x.quantile(0.25),
        #  '{}_user_{}_skew'.format(prefix, col): 'skew',

    }).reset_index()
    return group_df

def gen_user_diff_amount_features(df):
    group_df = df.groupby(['user'])['amount_per_time'].agg({
        'user_amount_diff_mean': 'mean',
       # 'user_amount_std': 'std',
        'user_amount_diff_max': 'max',
        'user_amount_diff_min': 'min',
        'user_amount_diff_sum': 'sum',
        'user_amount_diff_med': 'median',
        'user_amount_diff_cnt': 'count',
        # 'user_amount_q1': lambda x: x.quantile(0.25),
        # 'user_amount_q3': lambda x: x.quantile(0.75),
        }).reset_index()
    return group_df

def gen_user_group_amount_features(df, value):
    group_df = df.pivot_table(index='user',
                              columns=value,
                              values='amount',
                              dropna=False,
                              aggfunc=['count', 'sum',
                                       'mean', 'max', 'min', 'median',
                                       ]).fillna(0)
    group_df.columns = ['user_{}_{}_amount_{}'.format(value, f[1], f[0]) for f in group_df.columns]
    group_df.reset_index(inplace=True)

    return group_df

def gen_user_window_amount_features(df, window):
    group_df = df[df['days_diff']>window].groupby('user')['amount'].agg({
        'user_amount_mean_{}d'.format(window): 'mean',
        'user_amount_std_{}d'.format(window): 'std',
        'user_amount_max_{}d'.format(window): 'max',
        'user_amount_min_{}d'.format(window): 'min',
        'user_amount_sum_{}d'.format(window): 'sum',
        'user_amount_med_{}d'.format(window): 'median',
        'user_amount_cnt_{}d'.format(window): 'count',
        # 'user_amount_q1_{}d'.format(window): lambda x: x.quantile(0.25),
        # 'user_amount_q3_{}d'.format(window): lambda x: x.quantile(0.75),
        # 'user_amount_qsub_{}d'.format(window): lambda x: x.quantile(0.75) - x.quantile(0.25),
        # 'user_amount_skew_{}d'.format(window): 'skew',
        # 'user_amount_q4_{}d'.format(window): lambda x: x.quantile(0.8),
        # 'user_amount_q5_{}d'.format(window): lambda x: x.quantile(0.3),
        # 'user_amount_q6_{}d'.format(window): lambda x: x.quantile(0.7),
        }).reset_index()
    return group_df

def gen_user_hour_window_amount_features(df, start, end):
    group_df = df[(df['hour'] > start) & (df['hour'] <= end)].groupby('user')['amount'].agg({
        'user_hour_{}_to_{}_amount_mean'.format(start, end): 'mean',
        'user_hour_{}_to_{}_amount_std'.format(start, end): 'std',
        'user_hour_{}_to_{}_amount_max'.format(start, end): 'max',
        'user_hour_{}_to_{}_amount_min'.format(start, end): 'min',
        'user_hour_{}_to_{}_amount_sum'.format(start, end): 'sum',
        'user_hour_{}_to_{}_amount_med'.format(start, end): 'median',
        'user_hour_{}_to_{}_amount_cnt'.format(start, end): 'count',
    }).reset_index()

    return  group_df


def gen_user_nunique_features(df, value, prefix):
    group_df = df.groupby(['user'])[value].agg({
        'user_{}_{}_nuniq'.format(prefix, value): 'nunique'
    }).reset_index()
    return group_df

def gen_user_null_features(df, value, prefix):
    df['is_null'] = 0
    df.loc[df[value].isnull(), 'is_null'] = 1

    group_df = df.groupby(['user'])['is_null'].agg({'user_{}_{}_null_cnt'.format(prefix, value): 'sum',
                                                    'user_{}_{}_null_ratio'.format(prefix, value): 'mean'}).reset_index()
    return group_df

def gen_file_type_days_diff(df, file, type, time_feat):
    plot_feats = []
    #file_type_unique = file[type].value_counts().index.tolist()
    file_type_unique = []

    if type == 'type1':
        file_type_unique = ['45a1168437c708ff',
                             'f67d4b5a05a1352a', 
                             ]
    elif type == 'type2':
        file_type_unique = ['11a213398ee0c623',]

    elif type == 'channel':
        file_type_unique = ['b2e7fa260df4998d',
                            '116a2503b987ea81',
                            '8adb3dcfea9dcf5e']

    elif type == 'tunnel_io':
        file_type_unique = ['b2e7fa260df4998d_6ee790756007e69a',]
    elif type == 'type':
        file_type_unique = ['f67d4b5a05a1352a_nan',
                            '19d44f1a51919482_11a213398ee0c623',
                            '45a1168437c708ff_11a213398ee0c623',
                            '674e8d5860bc033d_11a213398ee0c623',
                            '0a3cf8dac7dca9d1_b5a8be737a50b171']

    for tp in file_type_unique:
        assert file_type_unique != []
        group_df = file[file[type] == tp].groupby(['user'])[time_feat].agg(
            {'user_{}_{}_min_{}'.format(type, tp, time_feat): 'min',
             'user_{}_{}_mean_{}'.format(type, tp, time_feat): 'mean',
             'user_{}_{}_max_{}'.format(type, tp, time_feat): 'max',
             'user_{}_{}_std_{}'.format(type, tp, time_feat): 'std',
             'user_{}_{}_median_{}'.format(type, tp, time_feat): 'median',
             'user_{}_{}_sum_{}'.format(type, tp, time_feat): 'sum',
             # 'user_{}_{}_q1_{}'.format(type, tp, time_feat): lambda x: x.quantile(0.25),
             # 'user_{}_{}_q3_{}'.format(type, tp, time_feat): lambda x: x.quantile(0.75),
             # 'user_{}_{}_q_sub_{}'.format(type, tp, time_feat): lambda x: x.quantile(0.75) - x.quantile(0.25),
             # 'user_{}_{}_skew_{}'.format(type, tp, time_feat): 'skew',
             }).reset_index()
        df = df.merge(group_df, on=['user'], how='left')
        stastic = ['min', 'max', 'max', 'std', 'median', 'sum',]
        for stast in stastic:
            plot_feats.append('user_{}_{}_{}_{}'.format(type, tp, stast, time_feat))

    return df, plot_feats

def gen_cnt_feature(df, feature):
    cnt_features = []
    for fea in feature:
        df[fea + '_count'] = df.groupby([fea])['user'].transform('count')
        cnt_features.append(fea + '_count')

    return df


def file_cols_user_nunique(file, feature_lst, prefix):
    col_nuniq_fea_lst = []
    for col in tqdm(feature_lst):
        col_nuniq = file.groupby(col)['user'].nunique()
        col_nuniq_dic = dict(zip(col_nuniq.index, col_nuniq.values))
        file[prefix + '_' + col + '_user_nuniq'] = file[col].map(col_nuniq_dic)
        col_nuniq_fea_lst.append(prefix + '_' + col + '_user_nuniq')

    return file, col_nuniq_fea_lst


def gen_stastic_col_user_nunique(file, feat, prefix):
    group_df = file.groupby('user')[feat].agg({
        prefix + feat + '_mean': 'mean',
        prefix + feat + '_std': 'std',
        prefix + feat + '_max': 'max',
        prefix + feat + '_min': 'min',
        prefix + feat + '_sum': 'sum',
        prefix + feat + '_med': 'median',
        #prefix + feat + '_q1'  : lambda x: x.quantile(0.25),
        #prefix + feat + '_q3'  : lambda x: x.quantile(0.75),
        #prefix + feat + 'q_sub': lambda x: x.quantile(0.75) - x.quantile(0.25),
        #prefix + feat + '_skew': 'skew',
    })
    return group_df

def gen_user_feat_cnt(file, col):
    group_df = file.groupby('user')[col].count()
    return group_df


def gen_features(df, op, trans, trans_op, window, num):
    df.drop(['service3_level'], axis=1, inplace=True)
    # base
    # df = fea_combine(df)
    # df = gen_fea_base_df(df)
    # base
    train_base = pd.read_csv('./data' + '/train/train_base.csv')
    int_cols = [col for col in train_base.columns if train_base[col].dtype == 'int64']
    df, cross_feature = int_cols_cross(df, int_cols)
    df['product7_fail_ratio'] = df['product7_fail_cnt'] / df['product7_cnt']
    cnt_feature = ['city', 'province', 'balance', 'ip_cnt', 'using_time', ]
    df = gen_cnt_feature(df, cnt_feature)

    # trans
    df = df.merge(gen_user_amount_features(trans), on=['user'], how='left')
    for col in tqdm(
            ['days_diff', 'platform', 'tunnel_in', 'tunnel_out', 'type1', 'type2', ]):  # 'trans_ip', 'trans_ip_3']):
        df = df.merge(gen_user_nunique_features(df=trans, value=col, prefix='trans'), on=['user'], how='left')

    trans_cols = ['days_diff', 'platform', 'tunnel_in', 'tunnel_out', 'type1', 'type2', ]
    trans, trans_col_nuniq_fea_lst = file_cols_user_nunique(trans, trans_cols, prefix='trans')
    for col in tqdm(trans_col_nuniq_fea_lst, desc='extract trans col user nunique stactic feature'):
        df = df.merge(gen_stastic_col_user_nunique(trans, col, 'trans'), on=['user'], how='left')

    df['user_amount_per_days'] = df['user_amount_sum'] / df['user_trans_days_diff_nuniq']
    df['user_amount_per_cnt'] = df['user_amount_sum'] / df['user_amount_cnt']
    df = df.merge(gen_user_group_amount_features(df=trans, value='platform'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(df=trans, value='type1'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(df=trans, value='type2'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(df=trans, value='type'), on=['user'], how='left')
    # df = df.merge(gen_user_group_amount_features(df=trans, value='type '), on=['user'], how='left')

    df = df.merge(gen_user_window_amount_features(df=trans, window=window[2]), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=window[1]), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=window[0]), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=10), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=5), on=['user'], how='left')
    # df = df.merge(gen_user_window_amount_features(df=trans, window=5), on=['user'], how='left')
    df = df.merge(gen_user_null_features(df=trans, value='trans_ip', prefix='trans'), on=['user'], how='left')
    df = df.merge(gen_user_null_features(df=trans, value='trans_ip_3', prefix='trans'), on=['user'], how='left')

    df, plot_feats = gen_file_type_days_diff(df, trans, 'type1', 'days_diff')
    df, _ = gen_file_type_days_diff(df, trans, 'type1', 'hour')
    df, _ = gen_file_type_days_diff(df, trans, 'type2', 'days_diff')
    df, _ = gen_file_type_days_diff(df, trans, 'type2', 'hour')
    df, _ = gen_file_type_days_diff(df, trans, 'tunnel_io', 'days_diff')
    df, _ = gen_file_type_days_diff(df, trans, 'tunnel_io', 'hour')

    #     df, _ = gen_file_type_days_diff(df, op, 'channel', 'days_diff')
    #     df, _ = gen_file_type_days_diff(df, op, 'channel', 'hour')

    df = df.merge(d2v_feat(df=trans, feat='amount', length=10, num=num), on=['user'], how='left', )
    df = df.merge(w2v_feat(df=trans, feat='amount', length=10, num=num), on=['user'], how='left', )

    # df = gen_trans_type_days_diff(df, trans, 'type1')
    df = df.merge(gen_user_tfidf_features(df=trans, value='platform_tunnel_io_type'), on=['user'], how='left')
    # #df = df.merge(gen_user_tfidf_features(df=op, value='net_type_channel'), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=trans, value='platform_tunnel_io_type'), on=['user'], how='left')
    time_cols = ['days_diff', 'week', 'hour', 'timestamp', ]
    for col in time_cols:
        df = df.merge(gen_stastic_col_user_nunique(trans, col, 'trans'), on=['user'], how='left')

    # df = df.merge(gen_user_countvec_features(df=trans, value='tunnel_out'), on=['user'], how='left')
    #     for wd in range(7):
    #         df = df.merge(gen_user_week_day_cnt(trans,'trans', wd), on=['user'], how='left')
    #     time_period = [-1, 8, 12, 15, 24]
    #     for tp in range(4):
    #         df = df.merge(gen_hour_period(trans, 'trans', time_period, tp,), on=['user'], how='left')

    # op
    df = df.merge(gen_user_tfidf_features(df=op, value='op_mode', ), on=['user'], how='left')
    df = df.merge(gen_user_tfidf_features(df=op, value='op_type', ), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_mode', ), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_type', ), on=['user'], how='left')
    op_cols = [col for col in op.columns if col not in ['user', 'tm_diff', 'timestmap', 'week']]
    op, op_col_nuniq_fea_lst = file_cols_user_nunique(op, op_cols, prefix='op')
    for col in tqdm(op_col_nuniq_fea_lst, desc='extract col user nunique stactic feature'):
        df = df.merge(gen_stastic_col_user_nunique(op, col, 'op'), on=['user'], how='left')

    for col in tqdm(op_cols, desc='extract nunique future'):
        df = df.merge(gen_user_nunique_features(df=op, value=col, prefix='op'), on=['user'], how='left')
    df = df.merge(w2v_feat(df=op, feat='channel', length=10, num=num), on=['user'], how='left')
    for col in time_cols:
        df = df.merge(gen_stastic_col_user_nunique(op, col, 'op'), on=['user'], how='left')
    #     for wd in range(7):
    #         df = df.merge(gen_user_week_day_cnt(op,'op', wd), on=['user'], how='left')
    #     for tp in range(4):
    #         df = df.merge(gen_hour_period(op, 'op', time_period, tp,), on=['user'], how='left')

    #     for col in tqdm(op_cols, desc='get op user feature mode'):
    #         df = df.merge(gen_user_feat_mode(op, col), on=['user'], how='left')
    # df = df.merge(w2v_feat(df=op, feat='op_ip', length=10,), on=['user'], how='left')
    # df = df.merge(w2v_feat(df=op, feat='op_ip_3', length=10), on=['user'], how='left')

    # df = df.merge(gen_user_countvec_features(df=op, value='op_device'), on=['user'], how='left')
    # df = df.merge(gen_user_tfidf_features(df=op, value='op_device'), on=['user'], how='left')
    # df = df.merge(gen_user_tfidf_features(df=op, value='op_ip_3'), on=['user'], how='left')
    # df = df.merge(gen_user_countvec_features(df=op, value='op_ip_3'), on=['user'], how='left')
    # trans_op
    trans_op_cols = [col for col in trans_op.columns if col not in ['user', 'tm_diff', 'timestmap', 'label', 'session']
                     + ['week_property', 'hour_property', 'day_property', 'day_week_property', 'day_hour_property']]

    trans_op, trans_op_col_nuniq_fea_lst = file_cols_user_nunique(trans_op, trans_op_cols, prefix='trans_op')
    for col in tqdm(trans_op_col_nuniq_fea_lst, desc='extract trans_op col user nunique stactic feature'):
        df = df.merge(gen_stastic_col_user_nunique(trans_op, col, 'trans_op'), on=['user'], how='left')

    for col in tqdm(trans_op_cols, desc='extract trans_op nunique future'):
        df = df.merge(gen_user_nunique_features(df=trans_op, value=col, prefix='trans_op'), on=['user'], how='left')

    df = df.merge(gen_user_group_trans_op_features(df=trans_op, columns='property', value='days_diff'), on=['user'],
                  how='left')
    #     df = df.merge(gen_user_group_trans_op_features(df=trans_op, columns='property', value='hour'), on=['user'], how='left')
    #     df = df.merge(gen_user_group_trans_op_features(df=trans_op, columns='property', value='week'), on=['user'], how='left')
    df = df.merge(gen_user_null_features(df=trans_op, value='trans_op_ip', prefix='trans_op'), on=['user'], how='left')
    df = df.merge(gen_user_null_features(df=trans_op, value='trans_op_ip_3', prefix='trans_op'), on=['user'],
                  how='left')
    for col in time_cols:
        df = df.merge(gen_stastic_col_user_nunique(op, col, 'trans_op'), on=['user'], how='left')
    #     group_dic = trans_op_df.groupby('user').apply(lambda x: session_time_delta(x)).to_dict()
    #     df['session_time_delta'] = df['user'].map(group_dic)
    #   dic = trans_op.groupby('user').apply(lambda x: if_ip_same(x)).to_dict()
    #     df['if_ip_same'] = df['user'].map(dic)
    df = user_trans_behavior_feature(df, trans_op)
    
    return df, cross_feature


def feature_supply(df, op, trans):
    # 添加交互特征
    interactive_fe = ['op1_cnt', 'card_a_cnt', 'op1_cnt', 'service1_cnt', 'service1_amt', 'agreement_total',
                      'acc_count', 'login_cnt_period1', 'ip_cnt', 'login_cnt_avg', 'login_days_cnt',
                      'product7_cnt', 'product7_fail_cnt']
    df, feature_cross_list = feature_cross(df, interactive_fe)

    # trans
    # print(pd.DataFrame(trans.groupby('platform')['user'].nunique()))
    # df = df.merge(gen_user_amount_features(trans), on=['user'], how='left')
    # trans = trans.merge(gen_user_amount_features(trans), on=['user'], how='left')
    # print(trans.columns)

    df = wsp_trans_feat(df, trans)
    # 对金额编码
    df = df.merge(gen_user_tfidf_features(df=trans, value='amount'), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=trans, value='amount'), on=['user'], how='left')

    # op
    df = df.merge(gen_user_tfidf_features(df=op, value='op_device'), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_device'), on=['user'], how='left')

    return df

def gen_fea_op_df(op_df):
    op_df['op_pattern'] = op_df['op_type'].map(str) + '_' + op_df['op_mode'].map(str) + '_' + op_df['op_device'].map(str)
    op_df['op_type_mode'] = op_df['op_type'].map(str) + '_' + op_df['op_mode'].map(str)
    op_df['op_type_device'] = op_df['op_type'].map(str) + '_' + op_df['op_device'].map(str)
    op_df['op_mode_device'] = op_df['op_mode'].map(str) + '_' + op_df['op_device'].map(str)
    op_df['ip_net_type'] = op_df['ip'].map(str) + '_' + op_df['net_type'].map(str)
    op_df['ip3_net_type'] = op_df['ip_3'].map(str) + '_' + op_df['net_type'].map(str)
    op_df['net_type_channel'] = op_df['net_type'].map(str) + '_' +  op_df['channel'].map(str)
   # op_df['time_diff'] = op_df['timestamp'].diff(-1)
    op_df.rename(columns={'ip' : 'op_ip', 'ip_3': 'op_ip_3',}, inplace=True)
    return op_df

def gen_fea_trans_df(trans_df):
    trans_df['tunnel_io'] = trans_df['tunnel_in'].astype(str) + '_' + trans_df['tunnel_out'].astype(str)
    trans_df['type'] = trans_df['type1'].astype(str) + '_' +trans_df['type2'].astype(str)
    trans_df['tunnel_io_type'] = trans_df['tunnel_io'].astype(str) + '_' + trans_df['type'].astype(str)
    trans_df['platform_tunnel_io_type'] = trans_df['platform'].astype(str) + '_' + trans_df['tunnel_io_type']
    trans_df['platform_tunnel_io'] = trans_df['platform'].astype(str) + '_' + trans_df['tunnel_io']
    trans_df['platform_type'] = trans_df['platform'].astype(str) + '_' + trans_df['type']
    trans_df['platform_amount'] = trans_df['platform'].astype(str) + '_' + trans_df['amount'].astype(str)
    trans_df['type_amount'] = trans_df['type'].astype(str) + '_' + trans_df['amount'].astype(str)
    trans_df['tunnel_io_amount'] = trans_df['type'].astype(str) + '_' + trans_df['amount'].astype(str)
    trans_df['type1_amount'] = trans_df['type1'].astype(str) + '_' + trans_df['amount'].astype(str)
    trans_df['type2_amount'] = trans_df['type2'].astype(str) + '_' + trans_df['amount'].astype(str)
    trans_df['tunnel_in_amount'] = trans_df['tunnel_in'].astype(str) + '_' + trans_df['amount'].astype(str)
    trans_df['tunnel_out_amount'] = trans_df['tunnel_out'].astype(str) + '_' + trans_df['amount'].astype(str)
    trans_df['amount_diff'] = trans_df['amount'].astype(int).diff(-1)
    trans_df['time_diff'] = trans_df['timestamp'].diff(-1)
    trans_df['amount_per_time'] = trans_df['amount_diff'] / np.where(trans_df['time_diff'] == 0, 0.01, trans_df['time_diff'])
    trans_df = gen_session_fea(trans_df)


    trans_df.rename(columns={'ip': 'trans_ip', 'ip_3': 'trans_ip_3',}, inplace=True)

    return  trans_df






