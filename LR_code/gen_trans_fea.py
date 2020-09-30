#/home/lirui/anaconda/envs python3.6.5
# -*- coding: UTF-8 -*-
"""
@Author: LiRui
@Date: 2020/8/18 上午11:40
@LastEditTime: 2020-09-02 23:15:51
@Description: 
@File: gen_trans_fea.py

"""
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold ,KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import gc
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def mode_count(x):
    try:
        return x.value_counts().iloc[0]
    except Exception:
        return np.nan

def data_group_pair(data, col, name):
    col2 = '_'.join(col)
    df = pd.DataFrame(data.groupby(col)['user'].nunique())
    df.columns = [name + 'cnt_user_' + col2]
    df[name + 'early_day_' + col2] = data.groupby(col)['days_diff'].min()
    df[name + 'later_day_' + col2] = data.groupby(col)['days_diff'].max()
    df[name + 'range_day_' + col2] = df[name + 'later_day_' + col2] - df[name + 'early_day_' + col2]

    df = df.reset_index()
    return df


def data_group(data, col, name):
    # day
    df = pd.DataFrame(data.groupby(col)['user'].nunique())
    df.columns = [name + 'cnt_user_' + col]
    # print(df)

    df[name + 'early_day_' + col] = data.groupby(col)['days_diff'].min()
    df[name + 'later_day_' + col] = data.groupby(col)['days_diff'].max()
    df[name + 'range_day_' + col] = df[name + 'later_day_' + col] - df[name + 'early_day_' + col]

    if name != 'trans':
        if col != 'op_device':
            df['op_d' + col] = data.groupby(col)['op_device'].apply(mode_count)

    df = df.reset_index()

    return df

def wsp_trans_feat(df, trans):
    print('wsp_trans_feat begin.......')
    # 时间特征
    data_tmp = pd.DataFrame(trans.groupby(['user'])['days_diff'].min().reset_index())
    data_tmp.columns = ['user', 'trans_user_early_day']
    df = df.merge(data_tmp, on=['user'], how='left')
    data_tmp = pd.DataFrame(trans.groupby(['user'])['days_diff'].max().reset_index())
    data_tmp.columns = ['user', 'trans_user_max_day']
    df = df.merge(data_tmp, on=['user'], how='left')
    # print(df[['user_early_day', 'user_max_day']])
    df['user_range_day'] = df['trans_user_max_day'] - df['trans_user_early_day']
    for wd in range(7):
        data_tmp = trans[trans['week'] == wd].groupby('user')['days_diff'].count().reset_index()
        data_tmp = pd.DataFrame(data_tmp)
        data_tmp.columns = ['user', 'trans_user_week_{}_cnt'.format(wd)]
        df = df.merge(data_tmp, on=['user'], how='left')

    time_period = [-1, 8, 12, 15, 23]
    for tp in range(4):
        data_tmp = pd.DataFrame(trans[((trans['hour'] > time_period[tp]) & (trans['hour'] < time_period[tp + 1]))]. \
                                groupby('user')['days_diff'].count().reset_index())
        data_tmp.columns = ['user', 'trans_user_time_period_{}_cnt'.format(tp)]
        df = df.merge(data_tmp, on=['user'], how='left')

    # 金额特征
    df['user_amount_range'] = df['user_amount_max'] - df['user_amount_min']

    # user关联到得设备，IP信息
    relate_var = ['platform', 'tunnel_in', 'tunnel_out', 'type1', 'type2', 'trans_ip', 'trans_ip_3']
    relate_pair = ['platform', 'type1', 'type2', 'trans_ip', 'trans_ip_3']

    for rv in relate_var:
        print('waiting for generating feature of {} ...'.format(rv))
        sample_data = trans[['user', rv]].drop_duplicates()
        group_data = data_group(trans, rv, 'trans')
        sample_data = sample_data.merge(group_data, on=rv, how='left')
        data_tmp = pd.DataFrame(sample_data.groupby('user')['trans' + 'cnt_user_' + rv].max())
        data_tmp.columns = ['relate_cnt_user_' + rv + '_max']
        data_tmp['relate_cnt_user_' + rv + '_min'] = sample_data.groupby('user')['trans' + 'cnt_user_' + rv].min()
        data_tmp['relate_cnt_user_' + rv + '_mean'] = sample_data.groupby('user')['trans' + 'cnt_user_' + rv].mean()
        data_tmp['relate_cnt_user_' + rv + '_skew'] = sample_data.groupby('user')['trans' + 'cnt_user_' + rv].skew()

        data_tmp['relate_early_day_' + rv + '_max'] = sample_data.groupby('user')['trans' + 'early_day_' + rv].max()
        data_tmp['relate_early_day_' + rv + '_min'] = sample_data.groupby('user')['trans' + 'early_day_' + rv].min()
        data_tmp['relate_early_day_' + rv + '_mean'] = sample_data.groupby('user')['trans' + 'early_day_' + rv].mean()
        data_tmp['relate_early_day_' + rv + '_skew'] = sample_data.groupby('user')['trans' + 'early_day_' + rv].skew()

        data_tmp['relate_later_day_' + rv + '_max'] = sample_data.groupby('user')['trans' + 'later_day_' + rv].max()
        data_tmp['relate_later_day_' + rv + '_min'] = sample_data.groupby('user')['trans' + 'later_day_' + rv].min()
        data_tmp['relate_later_day_' + rv + '_mean'] = sample_data.groupby('user')['trans' + 'later_day_' + rv].mean()
        data_tmp['relate_later_day_' + rv + '_skew'] = sample_data.groupby('user')['trans' + 'later_day_' + rv].skew()

        data_tmp['relate_range_day_' + rv + '_max'] = sample_data.groupby('user')['trans' + 'range_day_' + rv].max()
        data_tmp['relate_range_day_' + rv + '_min'] = sample_data.groupby('user')['trans' + 'range_day_' + rv].min()
        data_tmp['relate_range_day_' + rv + '_mean'] = sample_data.groupby('user')['trans' + 'range_day_' + rv].mean()
        data_tmp['relate_range_day_' + rv + '_skew'] = sample_data.groupby('user')['trans' + 'range_day_' + rv].skew()

        data_tmp = data_tmp.reset_index()
        # print(data_tmp.columns)
        df = df.merge(data_tmp, on=['user'], how='left')

    # for rv in combinations(relate_pair, 2):
    #     print('waiting for group pair feature of {} ...'.format(rv))
    #     rv2 = '_'.join(rv)
    #     sample_data = trans[['user'] + list(rv)].drop_duplicates()
    #     group_data = data_group_pair(trans, rv, 'trans')
    #     sample_data = sample_data.merge(group_data, on=rv, how='left')
    #
    #     data_tmp = pd.DataFrame(sample_data.groupby('user')['trans' + 'cnt_user_' + rv2].max())
    #     data_tmp.columns = ['relate_cnt_user_' + rv2 + '_max']
    #     data_tmp['relate_cnt_user_' + rv2 + '_min'] = sample_data.groupby('user')['trans' + 'cnt_user_' + rv2].min()
    #     data_tmp['relate_cnt_user_' + rv2 + '_mean'] = sample_data.groupby('user')['trans' + 'cnt_user_' + rv2].mean()
    #     data_tmp['relate_cnt_user_' + rv2 + '_skew'] = sample_data.groupby('user')['trans' + 'cnt_user_' + rv2].skew()
    #
    #     data_tmp['relate_early_day_' + rv2 + '_max'] = sample_data.groupby('user')['trans' + 'early_day_' + rv2].max()
    #     data_tmp['relate_early_day_' + rv2 + '_min'] = sample_data.groupby('user')['trans' + 'early_day_' + rv2].min()
    #     data_tmp['relate_early_day_' + rv2 + '_mean'] = sample_data.groupby('user')['trans' + 'early_day_' + rv2].mean()
    #     data_tmp['relate_early_day_' + rv2 + '_skew'] = sample_data.groupby('user')['trans' + 'early_day_' + rv2].skew()
    #
    #     data_tmp['relate_later_day_' + rv2 + '_max'] = sample_data.groupby('user')['trans' + 'later_day_' + rv2].max()
    #     data_tmp['relate_later_day_' + rv2 + '_min'] = sample_data.groupby('user')['trans' + 'later_day_' + rv2].min()
    #     data_tmp['relate_later_day_' + rv2 + '_mean'] = sample_data.groupby('user')['trans' + 'later_day_' + rv2].mean()
    #     data_tmp['relate_later_day_' + rv2 + '_skew'] = sample_data.groupby('user')['trans' + 'later_day_' + rv2].skew()
    #
    #     data_tmp['relate_range_day_' + rv2 + '_max'] = sample_data.groupby('user')['trans' + 'range_day_' + rv2].max()
    #     data_tmp['relate_range_day_' + rv2 + '_min'] = sample_data.groupby('user')['trans' + 'range_day_' + rv2].min()
    #     data_tmp['relate_range_day_' + rv2 + '_mean'] = sample_data.groupby('user')['trans' + 'range_day_' + rv2].mean()
    #     data_tmp['relate_range_day_' + rv2 + '_skew'] = sample_data.groupby('user')['trans' + 'range_day_' + rv2].skew()
    #
    #     data_tmp = data_tmp.reset_index()
    #     df = df.merge(data_tmp, on=['user'], how='left')

    return df


def judge_has_trans(df):
    operation = df['property'].unique()
    return True if 'trans' in operation else False


def last_trans_time(df):
    try:
        return df[df['property'] == 'trans']['days_diff'].values[-1]
    except Exception:
        return np.nan


def first_trans_time(df):
    try:
        return df[df['property'] == 'trans']['days_diff'].values[0]
    except Exception:
        return np.nan


def first_trans_hour(df):
    try:
        return df[df['property'] == 'trans']['hour'].values[0]
    except Exception:
        return np.nan


def last_trans_hour(df):
    try:
        return df[df['property'] == 'trans']['hour'].values[-1]
    except Exception:
        return np.nan


def first_trans_week(df):
    try:
        return df[df['property'] == 'trans']['week'].values[0]
    except Exception:
        return np.nan


def last_trans_week(df):
    try:
        return df[df['property'] == 'trans']['week'].values[-1]
    except Exception:
        return np.nan


def last_trans_timestamp(df):
    try:
        return df[df['property'] == 'trans']['timestamp'].values[-1]
    except Exception:
        return np.nan


def first_trans_timestamp(df):
    try:
        return df[df['property'] == 'trans']['timestamp'].values[0]
    except Exception:
        return np.nan


# 获取每个用户的交易次数
def gen_trans_count(df):
    df_ = df[df['property'] == 'trans'].copy()
    try:
        return df_['session'].nunique()
    except Exception:
        return np.nan


# 获取每个用户的交易次数
def gen_op_count(df):
    df_ = df[df['property'] == 'op'].copy()
    try:
        return df_['session'].nunique()
    except Exception:
        return np.nan


def user_trans_behavior_feature(df, trans_op):
    print("Starting extract user's trans behavior......")

    # 获取第一次和最后一次行为
    group_dic = trans_op.groupby('user').apply(lambda x: x['property'].values[-1]).to_dict()
    df['last_beahvior'] = df['user'].map(group_dic)
    group_dic = trans_op.groupby('user').apply(lambda x: x['property'].values[0]).to_dict()
    df['first_beahvior'] = df['user'].map(group_dic)
    # 是否有过交易行为
    group_dic = trans_op.groupby('user').apply(lambda x: judge_has_trans(x)).to_dict()
    df['has_trans'] = df['user'].map(group_dic)
    # 最后一次交易days_diff
    group_dic = trans_op.groupby('user').apply(lambda x: last_trans_time(x)).to_dict()
    df['last_days_diff_trans'] = df['user'].map(group_dic)
    # 第一次交易days_diff
    group_dic = trans_op.groupby('user').apply(lambda x: first_trans_time(x)).to_dict()
    df['first_days_diff_trans'] = df['user'].map(group_dic)
    # 最后一次交易hour
    group_dic = trans_op.groupby('user').apply(lambda x: last_trans_hour(x)).to_dict()
    df['last_hour_trans'] = df['user'].map(group_dic)
    # 第一次交易hour
    group_dic = trans_op.groupby('user').apply(lambda x: first_trans_hour(x)).to_dict()
    df['first_hour_trans'] = df['user'].map(group_dic)
    # 最后一次交易week
    group_dic = trans_op.groupby('user').apply(lambda x: last_trans_week(x)).to_dict()
    df['last_week_trans'] = df['user'].map(group_dic)
    # 第一次交易week
    group_dic = trans_op.groupby('user').apply(lambda x: first_trans_week(x)).to_dict()
    df['first_week_trans'] = df['user'].map(group_dic)
    # 最后一次交易timestamp
    group_dic = trans_op.groupby('user').apply(lambda x: last_trans_timestamp(x)).to_dict()
    df['last_time_trans'] = df['user'].map(group_dic)
    # 第一次交易timestamp
    group_dic = trans_op.groupby('user').apply(lambda x: first_trans_timestamp(x)).to_dict()
    df['first_time_trans'] = df['user'].map(group_dic)
    # 平均交易次数
    group_dic = trans_op.groupby('user').apply(lambda x: gen_trans_count(x)).to_dict()
    df['trans_count'] = df['user'].map(group_dic)
    # 操作次数
    group_dic = trans_op.groupby('user').apply(lambda x: gen_op_count(x)).to_dict()
    df['op_count'] = df['user'].map(group_dic)

    return df


def col_to_amount(trans, stat_lst=['mean', 'std']):
    columns = ['platform', 'tunnel_in', 'tunnel_out', 'tunnel_io', 'type1', 'type2', 'type',
               'days_diff', 'week', 'hour', 'tunnel_io_type', 'platform_type']
    for col in columns:
        for stat in stat_lst:
            trans[f'amount_to_{stat}_{col}'] = trans.groupby([col])['amount'].transform(stat)
    amount_to_cols = [col for col in trans.columns if 'amount_to' in col]

    return trans, amount_to_cols

def amount_trans_meta(df, trans, cols, agg_lst):
    new_meta = []
    for col in cols:
        tmp = trans.groupby(col)['amount'].agg('mean')
        trans['{}_amt_mean'.format(col)] = trans[col].map(tmp)
        trans['{}_amt_sub_rt'.format(col)] = trans['{}_amt_mean'.format(col)] - trans['amount']
        new_meta.append('{}_amt_sub_rt'.format(col))

    new_features = []
    for f in new_meta:
        tmp = trans.groupby('user')[f].agg(agg_lst).reset_index()
        tmp.columns = ['user'] + ['{}_{}'.format(f, agg_) for agg_ in agg_lst]
        df = df.merge(tmp, on='user', how='left')
        new_features +=['{}_{}'.format(f, agg_) for agg_ in agg_lst]

    return df

def df_time_ratio(df, start, end):
    try:
        return len(df[(df['hour'] > start) & (df['hour'] <= end)]) / len(df)
    except Exception:
        return 0

def df_weekend_ratio(df, ):
    try:
        return len(df[(df['week'] == 0) & (df['week'] == 6)]) / len(df)
    except Exception:
        return 0

def trans_hour_ratio(df, trans,):
    group_dic = trans.groupby(['user']).apply(lambda x: df_time_ratio(x, 0, 6)).to_dict()
    df['0_6_trans_ratio'] = df['user'].map(group_dic)
    group_dic = trans.groupby(['user']).apply(lambda x: df_time_ratio(x, 6, 12)).to_dict()
    df['6_12_trans_ratio'] = df['user'].map(group_dic)
    group_dic = trans.groupby(['user']).apply(lambda x: df_time_ratio(x, 12, 18)).to_dict()
    df['12_18_trans_ratio'] = df['user'].map(group_dic)
    group_dic = trans.groupby(['user']).apply(lambda x: df_time_ratio(x, 18, 24)).to_dict()
    df['18_24_trans_ratio'] = df['user'].map(group_dic)

    return df

def trans_weekend_ratio(df, trans):
    group_dic = trans.groupby(['user']).apply(lambda x: df_weekend_ratio(x)).to_dict()
    df['weekend_trans_ratio'] = df['user'].map(group_dic)

    return df



def gen_user_session_amount_features(trans, session):
    group_df = trans[trans['session'] == session].groupby(['user'])['amount'].agg({
        'user_amount_mean_session{}'.format(session): 'mean',
        'user_amount_std_session{}'.format(session): 'std',
        'user_amount_cnt_session{}'.format(session): 'count',
        'user_amount_med_session{}'.format(session): 'median',
        'user_amount_max_session{}'.format(session): 'max',
        'user_amount_min_session{}'.format(session): 'min',
        'user_amount_sum_session{}'.format(session): 'sum',

    }).reset_index()

    return group_df