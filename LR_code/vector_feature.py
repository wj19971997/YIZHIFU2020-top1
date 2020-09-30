#/home/lirui/anaconda/envs python3.6.5
# -*- coding: UTF-8 -*-
"""
@Author: LiRui
@Date: 2020/8/10 下午8:19
@LastEditTime: 2020-09-03 23:15:51
@Description: 
@File: vector_feature.py

"""
import pandas as pd
from gensim.models import Word2Vec
import os
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold ,KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

def gen_user_tfidf_features(df, value,):
    print('Start tfdif encoding for {}........'.format(value))
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['user', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = TfidfVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_tfidf_{}_{}'.format(value, i) for i in range(10)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df

def gen_user_countvec_features(df, value,):
    print('Start countvec encoding for {}........'.format(value))
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['user', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = CountVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_countvec_{}_{}'.format(value, i) for i in range(10)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df

def w2v_feat(df, feat, length, num):
    """

    :param df:         进行word2vec编码的数据
    :param feat:       进行编码的特征
    :param length:     embedding向量长度
    :return:
    """
    global w2v_fea_lst
    w2v_fea_lst = []
    print('Start training Word2Vec models.....')
    df[feat] = df[feat].astype(str)
    group_df = df.groupby(['user'])[feat].agg(list).reset_index()
    model = Word2Vec(group_df[feat].values, size=length, window=10, min_count=1, sg=1, hs=1, workers=1,
                     iter=20, seed=2020,)
    # if feat == 'amount':
    #     model = Word2Vec.load('../w2v_models/w2v_testb_{}_{}.model'.format(feat, num))
    # elif feat == 'channel':
    #     model = Word2Vec.load('../w2v_models/w2v_channel_16.model')
    # elif feat == 'trans_op_ip' or feat == 'trans_op_ip_3':
    #     model = Word2Vec.load('../w2v_models/w2v_trans_op_2.model')
    # else:
    #     if not os.path.exists('../w2v_models/'):
    #         os.makedirs('../w2v_models/')
    # model = Word2Vec.load('./w2v_models/w2v_testb_{}_{}.model'.format(feat, num))
    model.save('../w2v_models/w2v_testb_{}_{}.model'.format(feat, num))

    group_df[feat] = group_df[feat].apply(lambda x: pd.DataFrame([model[c] for c in x]))
    for m in tqdm(range(length), desc='extract w2v {} statistic feature'.format(feat)):
        group_df['{}_w2v_{}_mean'.format(feat,m)] = group_df[feat].apply(lambda x: x[m].mean())
        # group_df['{}_w2v_{}_median'.format(feat, m)] = group_df[feat].apply(lambda x: x[m].median())
        # group_df['{}_w2v_{}_max'.format(feat, m)] = group_df[feat].apply(lambda x: x[m].max())
        # group_df['{}_w2v_{}_min'.format(feat, m)] = group_df[feat].apply(lambda x: x[m].min())
        # group_df['{}_w2v_{}_sum'.format(feat, m)] = group_df[feat].apply(lambda x: x[m].sum())
        # group_df['{}_w2v_{}_std'.format(feat, m)] = group_df[feat].apply(lambda x: x[m].std())


        w2v_fea_lst.append('{}_w2v_{}_mean'.format(feat,m))
    del group_df[feat]

    return group_df

def d2v_feat(df, feat, length, num):
    print('Start training Doc2Vec models.......')
    df[feat] = df[feat].astype(str)
    group_df = df.groupby(['user'])[feat].agg(list).reset_index()
    documents = [TaggedDocument(doc, [i]) for i, doc in zip(group_df['user'].values, group_df[feat])]
    model = Doc2Vec(documents, vector_size=length, window=10, min_count=1, workers=1, seed=2020,
                    epochs=20,  hs=1, )
    if not os.path.exists('./d2v_models/'):
        os.makedirs('./d2v_models/')
    model.save('../d2v_models/d2v_testb_{}.model'.format(num))
    # model = Doc2Vec.load('./d2v_models/d2v_testb_{}.model'.format(num))
    doc_df = group_df['user'].apply(lambda x: ','.join([str(i) for i in model[x]])).str.split(',', expand=True).apply(pd.to_numeric)
    doc_df.columns = ['{}_d2v_{}'.format(feat, i) for i in range(length)]

    return pd.concat([group_df[['user']], doc_df], axis=1)