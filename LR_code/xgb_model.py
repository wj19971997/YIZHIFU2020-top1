'''
Author: lirui
Date: 2020-09-03 15:11:30
LastEditTime: 2020-09-04 20:31:37
Description: In User Settings Edit
FilePath: /YIZHIFU_2020_Cloud/code/xgb_model.py
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold ,KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import numpy as np
import xgboost as xgb

def xgb_model(train, target, test, k, lr, seed):
    drop_feature = np.load('./data/drop_feature.npy')
    feats = [f for f in train.columns if f not in ['user', 'label'] + list(drop_feature)]
    print('Current num of features:', len(feats))
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof_probs = np.zeros(train.shape[0])
    output_preds = 0
    offline_score = []
    feature_importance_df = pd.DataFrame()

    parameters = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'min_child_weight': 5,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': lr,
        'seed': 2020,
        # 'tree_method':'gpu_hist',
        'tree_method': 'hist',
    }

    for i, (train_index, test_index) in enumerate(folds.split(train, target)):
        train_y, test_y = target[train_index], target[test_index]
        train_X, test_X = train[feats].iloc[train_index, :], train[feats].iloc[test_index, :]

        dtrain = xgb.DMatrix(train_X, train_y)
        dval = xgb.DMatrix(test_X, test_y)
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        xgb_model = xgb.train(parameters, dtrain, num_boost_round=5000, evals=watchlist, verbose_eval=100,
                              early_stopping_rounds=100)
        oof_probs[test_index] = xgb_model.predict(xgb.DMatrix(test_X),
                                                 ntree_limit=xgb_model.best_ntree_limit)
        print(xgb_model.best_score)
        offline_score.append(xgb_model.best_score)
        output_preds += xgb_model.predict(xgb.DMatrix(test[feats]),
                                          ntree_limit=xgb_model.best_ntree_limit
                                          ) / folds.n_splits
        print(offline_score)

    print('OOF-MEAN-AUC:%.6f, OOF-STD-AUC:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    print('valid-AUC:%.6f' % (roc_auc_score(target, oof_probs)))
    # oof_probs = pd.DataFrame(oof_probs, columns=['train_xgb_pre_{}_{}'.format(eta,t)])
    # oof_probs.to_csv('melt/melt_model_xgb_{}_{}.csv'.format(eta,t))

    return output_preds,  oof_probs, np.mean(offline_score),
