'''
Author: lirui
Date: 2020-08-31 12:39:18
LastEditTime: 2020-09-04 21:23:04
Description: 融合函数
FilePath: /Datacastle_2020YIZHIFU-master/code/merge.py
'''

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

#from GcForest_model import *
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold ,KFold
from sklearn.metrics import roc_auc_score
from LR_code.gen_base_fea import min_max_unif




# 数据读取
if __name__ == '__main__':
    print('数据读取....')


    DATA_PATH = './data'
    print('Loading data......')
    
    target = pd.read_csv(DATA_PATH + '/train/train_label.csv')['label']
    testb_base = pd.read_csv(DATA_PATH + '/test/testb_base.csv')
    #读取914特征模型线下和线上概率文件

    xgb_oof_914fea_1 = np.load('./xgb_seed2020/xgb_val_914fea.npy')
    xgb_preds_914fea_1 = np.load('./xgb_seed2020/xgb_ts_914fea.npy')
    xgb_oof_914fea_2 = np.load('./xgb_seed779/xgb_val_914fea.npy')
    xgb_preds_914fea_2 = np.load('./xgb_seed779/xgb_ts_914fea.npy')
    xgb_914_oof = xgb_oof_914fea_1 * 0.5 + xgb_oof_914fea_2 * 0.5
    xgb_914_preds = xgb_preds_914fea_1 * 0.5 + xgb_preds_914fea_2 * 0.5
    
    #读取LR 线下线上xgb概率文件
    xgb_oof_1 = np.load('./xgb_seed2020/xgb_val_1674fea.npy')
    xgb_preds_1 = np.load('./xgb_seed2020/xgb_ts_1674fea.npy')
    xgb_oof_2 = np.load('./xgb_seed2020/xgb_val_1690fea.npy')
    xgb_preds_2 = np.load('./xgb_seed2020/xgb_ts_1690fea.npy')
    xgb_oof_3 = np.load('./xgb_seed779/xgb_val_1674fea.npy')
    xgb_preds_3 = np.load('./xgb_seed779/xgb_ts_1674fea.npy')
    #LR xgb概率文件融合
    xgb_merge_oof = xgb_oof_1 * 0.33 + xgb_oof_2 * 0.33 + xgb_oof_3 * 0.33
    xgb_merge_preds = xgb_preds_1 * 0.33 + xgb_preds_2 * 0.33 + xgb_preds_3 * 0.33
    #读取WJ nn概率文件
    nn_wj_oof = np.load('./nn_prob/val.npy')
    nn_wj_preds = pd.read_csv('./nn_prob/test.csv')['prob'].values
    #三人模型融合(7:2:1)
    merge_oof = min_max_unif(xgb_merge_oof) * 0.7 + min_max_unif(nn_wj_oof) * 0.1 + min_max_unif(xgb_914_oof) * 0.2
      
    merge_preds = min_max_unif(xgb_merge_preds) * 0.7 + min_max_unif(nn_wj_preds) * 0.1 + min_max_unif(xgb_914_preds) * 0.2 
     
    #线下auc
    print('xgb single model loacl score: ', roc_auc_score(target.values, xgb_oof_2))
    print('xgb random num merge model loacl score: ', roc_auc_score(target.values, xgb_merge_oof))
    print('final version loacl score: ', roc_auc_score(target.values, merge_oof))
    #提交文件生成(xgb单模最高一版)
    sub = testb_base[['user']].copy()
    sub['prob'] = xgb_preds_2
    sub.to_csv('./submission/sub_testb_xgb.csv', index=False)
    #提交文件生成(xgb随机数融合)
    sub['prob'] = xgb_merge_preds
    sub.to_csv('./submission/sub_testb_xgb_mix3.csv', index=False)
    #提交文件生成(最终版)
    sub['prob'] = merge_preds
    sub.to_csv('./submission/sub_testb_mix_lr_wsp_wj_6_3_1_751783.csv', index=False)

    print(sub.head())