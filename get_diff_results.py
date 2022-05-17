import pickle
from pprint import pprint
from socket import VM_SOCKETS_INVALID_VERSION

import pandas as pd

import torch
import torch.nn.functional as F

def get_diff_norm(x, y):
    
    diff = torch.abs(x-y).mean(dim=0)
    diff_mean = torch.abs(x-y)
    diff_mean = torch.mean(diff_mean)
    base_mean = torch.mean(y)
    mean = torch.mean(x)
    diff_norm_L1 = torch.norm(diff, p=1)
    diff_norm_L2 = torch.norm(diff, p=2)
    base_norm_L1 = torch.norm(y.mean(dim=0), p=1)
    base_norm_L2 = torch.norm(y.mean(dim=0), p=2)
    return mean.item(), diff_mean.item(), diff_mean.div_(base_mean).item(), diff_norm_L1.item(), diff_norm_L2.item(), diff_norm_L1.div_(base_norm_L1).item(), diff_norm_L2.div_(base_norm_L2).item()


if __name__ == '__main__':

    device_list = ['gpu', 'tpu']
    op_list = ['exp', 'softmax', 'GeLU', 'tanh', 'sigmoid', 'swish']
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
            mean, diff_mean, base_ratio, diff_norm_L1, diff_norm_L2, diff_ratio_L1, diff_ratio_L2 = get_diff_norm(op_result_tensor, op_val['gpu'])
            print("*"*20)
            print(f"op_name : {op_name}, device : {device},mean : {mean}")
            item = {
                'op': op_name,
                'device': device,
                'diff_mean' : diff_mean,
                'base_ratio' : base_ratio,
                'diff_norm_L1': diff_norm_L1,
                'diff_norm_L2': diff_norm_L2,
                'diff_ratio_L1': diff_ratio_L1,
                'diff_ratio_L2': diff_ratio_L2,
                'mean' : mean
            }
            
            diff_result.append(item)


    result_pd = pd.DataFrame(diff_result)
    pprint(result_pd)
    
    df_diff_mean = pd.pivot(result_pd, index='op', columns='device', values='diff_mean')
    print("\n\nDIFF_MEAN\n")
    pprint(df_diff_mean)
    
    df_base_ratio = pd.pivot(result_pd, index='op', columns='device', values='base_ratio')
    print("\n\nBASE_RATIO\n")
    pprint(df_base_ratio)
    
    df_diff_norm_L1 = pd.pivot(result_pd, index='op', columns='device', values='diff_norm_L1')
    print("\n\n\nDIFF_NORM_L1\n")
    pprint(df_diff_norm_L1)
    
    df_diff_norm_L2 = pd.pivot(result_pd, index='op', columns='device', values='diff_norm_L2')
    print("\n\n\nDIFF_NORM_L2\n")
    pprint(df_diff_norm_L2) 
    
    df_diff_ratio_L1 = pd.pivot(result_pd, index='op', columns='device', values='diff_ratio_L1')
    print("\n\n\nDIFF_RAIO_L1\n")
    pprint(df_diff_ratio_L1) 
    
    df_diff_ratio_L2 = pd.pivot(result_pd, index='op', columns='device', values='diff_ratio_L2')
    print("\n\n\nDIFF_RAIO_L2\n")
    pprint(df_diff_ratio_L2) 
    
    df_mean = pd.pivot(result_pd, index='op', columns='device', values='mean')
    print("\n\nMEAN\n")
    pprint(df_mean)
    
    df_diff_mean.to_csv('./result/result_diff_mean.csv')
    df_base_ratio.to_csv('./result/resut_diff_ratio.csv')
    result_pd.to_csv('./result/result_table.csv')
    df_diff_norm_L1.to_csv('./result/result_norm_L1_table.csv')
    df_diff_norm_L2.to_csv('./result/result_norm_L2_table.csv')
    df_diff_ratio_L1.to_csv('./result/reslut_ratio_L1_table.csv')
    df_diff_ratio_L2.to_csv('./result/reslut_ratio_L2_table.csv')
    df_mean.to_csv('./result/mean.csv')