#/home/lirui/anaconda/envs python3.6.5
# -*- coding: UTF-8 -*-
"""
@Author: LiRui
@Date: 2020/8/9 下午4:47
@LastEditTime: 2020-07-06 23:15:51
@Description: 画特征分布图的可视化函数, 减小内存占用函数及定义全局随机种子的函数
@File: utils.py

"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

def feature_classify(feature, train, test):
    denseFealist = []
    sparseFealist = []
    for i in feature:

        if test[i].dtype == 'float64':
            denseFealist.append(i)
        elif test[i].dtype == 'int64':
            sparseFealist.append(i)
    n_Unique_1000list = [col for col in feature if train[col].nunique() < 1000
                        and test[col].dtype == 'int64']
    n_Unique_1000listdense = [col for col in feature if train[col].nunique() < 1000
                             and test[col].dtype == 'float64']

    return denseFealist, sparseFealist, n_Unique_1000list, n_Unique_1000listdense

def plot_feature(feature, train, test, feature_importance_df):

    denseFealist, sparseFealist, n_Unique_1000list, n_Unique_1000listdense = feature_classify(feature, train, test)
    plt.gcf()

    # for index,value in enumerate(features):
    for fea in feature:
        print('Feature name: ', fea, "Nunique: ", train[fea].nunique(), )
        print('Score: ', feature_importance_df[feature_importance_df.feature == fea].importance)
        if fea in denseFealist:
            if fea in n_Unique_1000listdense:
                sns.countplot(train[str(fea)],label=fea)
                sns.countplot(test[str(fea)],label=fea)
                plt.show()
                train[[fea, 'label']].groupby([fea]).mean().plot.bar()
                plt.show()
            # 画pdf图(整体)
            else:
                facet1 = sns.FacetGrid(train, aspect=4)
                facet1.map(sns.kdeplot, fea, shade=True)
                facet2 = sns.FacetGrid(test, aspect=4)
                facet2.map(sns.kdeplot, fea, shade=True)
                # 画pdf图(type)
                facet = sns.FacetGrid(train, hue='label', aspect=4)
                facet.map(sns.kdeplot, fea, shade=True)
                facet.add_legend()
                plt.show()

                # 画训练集箱型图
                train.boxplot(column=fea, by='label', showfliers=False)
                plt.show()
        elif fea in sparseFealist:
            sns.countplot(train[str(fea)],label=fea)
            sns.countplot(test[str(fea)],label=fea)
            plt.show()
            train[[fea, 'label']].groupby([fea]).mean().plot.bar()
            plt.show()

def reduce_mem_usage(df):
    """

    :param df:
    :description: reduce memory usage
    :return:
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage dataframe is {:.2f}'.format(start_mem))
    feature_lst = [col for col in df.columns if col not in ['user', 'label']]
    for col in feature_lst:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min() and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f}'.format(end_mem))
    print('Decreased by {:.1f}'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def seed_everything(seed=2020):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def cal_pro_badrate(df, col):
    group = df.groupby(col)
    df = pd.DataFrame()
    df['total'] = group['label'].count()
    df['bad'] = group['label'].sum()
    df['bad_rate'] = df['bad'] / df['total']
    print(df.sort_values('bad_rate', ascending=False).iloc[:5, :])
