#/home/lirui/anaconda/envs python3.6.5
# -*- coding: UTF-8 -*-
"""
@Author: LiRui
@Date: 2020/8/16 下午9:04
@LastEditTime: 2020-07-06 23:15:51
@Description: 
@File: trans_op_merge.py

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

def gen_session_fea(file):
    tmp = file.groupby('user')['timestamp'].shift()
    file['delta_time'] = file['timestamp'] - tmp
    file['session'] = np.where(file['delta_time'] > 600, 1, 0)
    file['session'] = file.groupby(['user'])['session'].transform('cumsum')
    del file['delta_time']

    return file

def gen_trans_op_label(trans, op, train_label):
    trans_ = trans.copy()
    op_ = op.copy()
    user_label_dict = dict(zip(train_label.user.values, train_label.label.values))
    trans_['label'] = trans_['user'].map(user_label_dict)
    op_['label'] = op_['user'].map(user_label_dict)

    return trans_, op_


def trans_op_merge(trans, op):

    trans['property'] = 'trans'
    op['property'] = 'op'

    common_cols = ['user', 'ip', 'ip_3', 'tm_diff', 'days_diff', 'hour', 'week', 'timestamp', 'property', 'label']
    trans_op = pd.concat([trans[common_cols], op[common_cols]], axis=0)
    trans_op.sort_values(by=['user', 'timestamp'], inplace=True)
    trans_op.reset_index(inplace=True, drop=True)

    return trans_op

def gen_fea_trans_op_df(trans_op,):

    trans_op['week_property'] = trans_op['week'].map(str) + '_' + trans_op['property'].map(str)

    trans_op['hour_property'] = trans_op['hour'].map(str) + '_' + trans_op['property'].map(str)
    trans_op['day_property'] = trans_op['days_diff'].map(str) + '_' + trans_op['property'].map(str)
    trans_op['day_week_property'] = trans_op['days_diff'].map(str) + '_' + trans_op['week'].map(str) +\
                                    '_' + trans_op['property'].map(str)
    trans_op['day_hour_property'] = trans_op['days_diff'].map(str) + '_' + trans_op['hour'].map(str) + \
                                    '_' + trans_op['property'].map(str)
    trans_op['time_diff'] = trans_op['timestamp'].diff(-1)
    trans_op = gen_session_fea(trans_op)
    trans_op.rename(columns={'ip': 'trans_op_ip', 'ip_3': 'trans_op_ip_3'}, inplace=True)

    return trans_op

def gen_user_group_trans_op_features(df, columns, value):
    group_df = df.pivot_table(index='user',
                              columns=columns,
                              values=value,
                              dropna=False,
                              aggfunc=['count', 'sum',
                                       'mean', 'max', 'min', 'median',
                                       ]).fillna(0)
    group_df.columns = ['user_{}_{}_{}_{}'.format(columns, f[1], value, f[0]) for f in group_df.columns]
    group_df['op_trans_ratio'] = group_df['user_property_trans_{}_count'.format(value)] / group_df[
        'user_property_op_{}_count'.format(value)]

    group_df.reset_index(inplace=True)

    return group_df


def dic_avg(dic):
    L = len(dic)
    S = sum(dic.values())
    A = S / L

    return A


def session_time_delta(df):
    session = df['session'].unique()
    time_dic = {}
    start = 0
    for sess in session:
        t = df[df['session'] == sess].iloc[-1].timestamp
        time_dic['sess_{}'.format(sess)] = t - start
        start = t
    time_ave = dic_avg(time_dic)

    return time_ave


def if_ip_same(df):
    trans_ip = df.loc[df.property == 'trans']['trans_op_ip'].unique().tolist()
    op_ip = df.loc[df.property == 'op']['trans_op_ip'].unique().tolist()
    cross = [ip for ip in trans_ip if ip in op_ip]
    cross = [val for val in cross if type(val) == str]

    return True if cross != [] else False


def if_ip3_same(df):
    trans_ip = df.loc[df.property == 'trans']['trans_op_ip_3'].unique().tolist()
    op_ip = df.loc[df.property == 'op']['trans_op_ip_3'].unique().tolist()
    cross = [ip for ip in trans_ip if ip in op_ip]
    cross = [val for val in cross if type(val) == str]

    return True if cross != [] else False

