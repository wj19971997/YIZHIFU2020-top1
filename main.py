'''
Author: lirui
Date: 2020-09-03 16:19:01
LastEditTime: 2020-09-04 20:13:33
Description: 主函数, 调用main_lr(), get_result(), 生成待融合的结果, 供merge.py进行融合
FilePath: /YIZHIFU_2020_Cloud/main.py
'''
from LR_code.main_LR import *
from WJ_code.get_nn_prediction import get_result
def get_para(seed):
    parser = argparse.ArgumentParser(description='help info')
    parser.add_argument('--FOLDS', default=5, type=int)
    parser.add_argument('--tfdif_size', default=10, type=int)
    parser.add_argument('--countvec_size', default=10, type=int)
    parser.add_argument('--num', default=1, type=int)
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--SEED', default=1116, type=int)
    args = parser.parse_args()
    print('\n')
    print('seed: ', args.seed)
    print('SEED: ', args.SEED)
    print('num: ', args.num)
    return args

if __name__ == '__main__':

    # get lr prediction
    args = get_para(seed=2020)
    main_lr(window=[15, 23, 27], num=args.num, seed=args.seed, trans_behavior_fea=True, args=args, )
    args = get_para(seed=779)
    main_lr(window=[15, 23, 27], num=args.num, seed=args.seed, trans_behavior_fea=False, args=args, )
    # # get nn prediction
    print('Start get nn prediction......')
    get_result(epoch_num=10, bs=512, sub_name='test.csv', test_name='testb')
    # get wsp prediction
    # main_lr(window=[15, 23, 27], num=args.num, seed=2020, trans_behavior_fea=False, args=args, drop_crossfea=True)
    # main_lr(window=[15, 23, 27], num=args.num, seed=9999, trans_behavior_fea=False, args=args, drop_crossfea=True)
