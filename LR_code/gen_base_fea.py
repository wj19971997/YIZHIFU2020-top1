#/home/lirui/anaconda/envs python3.6.5
# -*- coding: UTF-8 -*-
"""
@Author: LiRui
@Date: 2020/8/27 上午10:45
@LastEditTime: 2020-07-06 23:15:51
@Description: 
@File: gen_base_fea.py

"""
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import gc
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def gen_fea_base_df(df):

    #类别特征组合编码
    df['provider_regist'] = df['provider'].map(str) + '_' + df['regist_type'].map(str)
    # df['card_cnt'] = df['card_a_cnt'].map(str) + '_' + df['card_b_cnt'].map(str) + '_' + df['card_d_cnt'].map(str) + \
    #                  df['card_d_cnt'].map(str)
    df['level_verified'] = df['level'].map(str) + '_' + df['verified'].map(str)
    #df['op_cnt'] = df['op1_cnt'].map(str) + '_' + df['op2_cnt'].map(str)
    #df['service_cnt'] = df['service1_cnt'].map(str) + '_' + df['service2_cnt'].map(str)
    df['agreement'] = df['agreement1'].map(str) + '_' + df['agreement2'].map(str) + \
                      df['agreement3'].map(str) + '_' + df['agreement4'].map(str)
    #df['login_cnt_period'] = df['login_cnt_period1'].map(str) + '_' + df['login_cnt_period2'].map(str)
    df['province_city'] = df['province'].map(str) + '_' + df['city'].map(str)
    df['balance_all'] = df['balance1'].map(str) + '_' + df['balance2'].map(str)
    df['balance_avg_all'] = df['balance1_avg'].map(str) + '_' + df['balance2_avg'].map(str)
    df['product_amount'] = df['product1_amount'].map(str) + '_' + df['product2_amount'].map(str) + '_' + \
                           df['product3_amount'].map(str) + '_' + df['product4_amount'].map(str) + '_' + \
                           df['product5_amount'].map(str)
    """
    #特征交叉


    df['card_cnt_sum'] = df['card_a_cnt'].astype(int) + df['card_b_cnt'].astype(int) + \
                         df['card_c_cnt'].astype(int) + df['card_d_cnt'].astype(int)
    df['op_cnt_sum'] = df['op1_cnt'] + df['op2_cnt']
    # df['service1_amt_per'] = df['service1_amt'] / df['service1_cnt']
    # df['service_cnt_sub'] = df['service1_cnt'] - df['service2_cnt']
    df['service_cnt_sum'] = df['service1_cnt'] + df['service2_cnt']

    df['login_cnt_sum'] = df['login_cnt_period1'] + df['login_cnt_period2']
   # df['login_sum'] = df['login_cnt_avg'] * df['login_days_cnt']
    """
    return df


def min_max_unif(arr):

    return (arr - arr.min()) / arr.max()


def int_cols_cross(df, cols):
    """[summary]
        对base数据的int64特征进行min-max归一化后进行加减乘除交互
    Parameters
    ----------
    df : [DataFrame]
        [训练集和测试集合并的数据]
    cols : [list]
        [交互特征]

    Returns
    -------
    [DataFrame, list]
        [整数特征交互后的data, 及交互特征名称]
    """
    cross_feature = []
    df = df.copy()
    for i, col in tqdm(enumerate(cols), desc='extract cross feature for base'):
        for j in range(i + 1, len(cols)):
            df[col + '_' + 'div_' + cols[j]] = min_max_unif(df[col]) / min_max_unif(df[cols[j]])
            df[col + '_' + 'sub_' + cols[j]] = min_max_unif(df[col]) - min_max_unif(df[cols[j]])
            df[col + '_' + 'mul_' + cols[j]] = min_max_unif(df[col]) * min_max_unif(df[cols[j]])
            df[col + '_' + 'sum_' + cols[j]] = min_max_unif(df[col]) + min_max_unif(df[cols[j]])

            cross_feature.append(col + '_' + 'div_' + cols[j])
            cross_feature.append(col + '_' + 'sub_' + cols[j])
            cross_feature.append(col + '_' + 'mul_' + cols[j])
            cross_feature.append(col + '_' + 'sum_' + cols[j])

    return df, cross_feature

def province_binary(df, ):
    """[summary]
        对风险率排名最高的五个省份进行二值化及组合编码
    Parameters
    ----------
    df : [data数据]
        [train or test]

    Returns
    -------
    [DataFrame]
        [经过省份二值化编码的训练集/测试集]
    """
    #省份二值化编码
    df['is_21_province'] = df.apply(lambda x: 1 if x.province == 21 else 0, axis=1)
    df['is_26_province'] = df.apply(lambda x: 1 if x.province == 26 else 0, axis=1)
    df['is_30_province'] = df.apply(lambda x: 1 if x.province == 30 else 0, axis=1)
    df['is_20_province'] = df.apply(lambda x: 1 if x.province == 20 else 0, axis=1)
    df['is_16_province'] = df.apply(lambda x: 1 if x.province == 16 else 0, axis=1)

    df['binary_province'] = df['is_21_province'].map(str) + df['is_26_province'].map(str) + \
                               df['is_30_province'].map(str) + df['is_20_province'].map(str) + df['is_16_province'].map(str)

    le = LabelEncoder()
    df['binary_province'].fillna('-1', inplace=True)
    df['binary_province'] = le.fit_transform(df['binary_province'])

    return df

def city_binary(df, ):
    #city binary encode
    pass