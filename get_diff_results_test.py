import pickle
from pprint import pprint

import pandas as pd

import torch
import torch.nn.functional as F

def get_diff_norm(x, y):
    diff = torch.abs(x-y)
    ratio = torch.div(diff, x)
    ratio_mean = torch.mean(ratio)
    L1_norm = torch.norm(ratio, p=1)
    L2_norm = torch.norm(ratio, p=2)
    ratio_flatten = torch.flatten(ratio)
    ratio_sort = torch.sort(ratio_flatten, descending=True)[0]
    ratio_top5 = ratio_sort[:5]
    ratio_bottom5 = ratio_sort[-5:]
    
    
    
    return ratio_mean.item(), L1_norm.item(), L2_norm.item(), torch.Tensor.tolist(ratio_top5), torch.Tensor.tolist(ratio_bottom5)

if __name__ == '__main__':

    device_list = ['gpu', 'tpu']
    op_list = ['softmax', 'GeLU', 'tanh', 'swish']
    op_result = {}

    # loading previous results
    for op in op_list:
        op_result[op] = {}
        for device in device_list:
            with open(f"./tensors/{device}_output/{device}_torch_{op}_output.pickle", "rb") as fr:
                op_result[op][device] = pickle.load(fr)

    # calculate diff and norm
    
    diff_result = []
    for op_name, op_val in op_result.items(): # 이 밑에는 device별로 있음
        
        for device, op_result_tensor in op_val.items():
            ratio_mean, L1_norm, L2_norm, ratio_top5, ratio_bottom5 = get_diff_norm(op_result_tensor, op_val['gpu'])
            item = {
                'op': op_name,
                'device': device,
                'ratio_mean' : ratio_mean,
                'L1_norm' : L1_norm,
                'L2_norm' : L2_norm,
                'ratio_top5' : ratio_top5,
                'ratio_bottom5' : ratio_bottom5
            }
            
            diff_result.append(item)


    result_pd = pd.DataFrame(diff_result)
    pprint(result_pd)
    
    df_ratio_mean = pd.pivot(result_pd, index='op', columns='device', values='ratio_mean')
    print("\n\nratio_mean\n")
    pprint(df_ratio_mean)
    df_L1_norm = pd.pivot(result_pd, index='op', columns='device', values='L1_norm')
    print("\n\nL1_loss\n")
    pprint(df_L1_norm)
    df_L2_norm = pd.pivot(result_pd, index='op', columns='device', values='L2_norm')
    print("\n\nL1_loss\n")
    pprint(df_L2_norm)
    df_ratio_top5 = pd.pivot(result_pd, index='op', columns='device', values='ratio_top5')
    print("\n\nratio_top5")
    pprint(df_ratio_top5)
    df_ratio_bottom5 = pd.pivot(result_pd, index='op', columns='device', values='ratio_bottom5')
    print("\n\nratio_top5")
    pprint(df_ratio_bottom5)
    
    result_pd.to_csv('./result/-1~1/result.csv')
    df_ratio_mean.to_csv('./result/-1~1/result_ratio_mean.csv')
    df_L1_norm.to_csv('./result/-1~1/result_L1_norm.csv')
    df_L2_norm.to_csv('./result/-1~1/result_L2_norm.csv')
    df_ratio_top5.to_csv('./result/-1~1/result_ratio_top5.csv')
    df_ratio_bottom5.to_csv('./result/-1~1/result_ratio_bottom5.csv')