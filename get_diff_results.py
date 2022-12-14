import torch
import pickle
import pandas
import pandas as pd
from pprint import pprint

def get_diff_prob(x, y):
    diff = torch.abs(x-y)
    x_abs = torch.abs(x)
    ratio = torch.div(diff,x_abs)
    ratio_mean = torch.mean(ratio)
    ratio_flatten = torch.flatten(ratio)
    ratio_sort = torch.sort(ratio_flatten, descending=True)[0]
    ratio_top5 = ratio_sort[:5]
    ratio_bottom5 = ratio_sort[-5:]
    
    significant_figure = 6
    gpu_sf_dpoint = significant_figure - torch.floor(torch.log10(x_abs)) - 1
    gpu_sf_dpoint = torch.nan_to_num(gpu_sf_dpoint, nan=0.0)
    diff = torch.abs(x-y)
    diff_dpoint = -(torch.floor(torch.log10(diff)))
    diff_dpoint = torch.nan_to_num(diff_dpoint, nan=0.0)
    #print(f'(torch.log10(x)) : {(torch.log10(x))}, diff : {diff}, diff_dpoint : {diff_dpoint}')
    #print(f"gpu_sf_dpoint : {gpu_sf_dpoint}, deff_dpoint : {diff_dpoint}")
    
    
    device_error = torch.where(gpu_sf_dpoint-diff_dpoint>=0, 1, 0)
    device_error[diff_dpoint==0] = 0
    
    
    error_cnt = torch.count_nonzero(device_error)
    prob = error_cnt/torch.numel(x)
    return ratio_mean.item(), torch.Tensor.tolist(ratio_top5), torch.Tensor.tolist(ratio_bottom5), prob.item()


if __name__=='__main__':
    device_list = ['gpu', 'm1']
    op_list = ['softmax', 'GeLU', 'tanh', 'swish']
    op_result = {}

    for op in op_list:
        op_result[op] = {}
        for device in device_list:
            with open(f"./tensors/{device}_output/posit10/{device}_torch_{op}_output.pickle", "rb") as fr:
                op_result[op][device] = pickle.load(fr)

    diff_result = []            
    for op_name, op_val in op_result.items():
        for device, op_result_tensor in op_val.items():
            ratio_mean,ratio_top5, ratio_bottom5 , prob = get_diff_prob(op_val['gpu'], op_result_tensor)
            item = {
                'op' : op_name,
                'device' : device,
                'ratio_mean' : ratio_mean,
                'ratio_top5' : ratio_top5,
                'ratio_bottom5' : ratio_bottom5,
                'prob' : prob
            }
            
            diff_result.append(item)
            
    result_pd = pd.DataFrame(diff_result)
    pprint(result_pd)

    df_ratio_mean = pd.pivot(result_pd, index='op', columns='device', values='ratio_mean')
    print('\n\nratio_mean\n')
    pprint(df_ratio_mean)
    
    df_ratio_top5 = pd.pivot(result_pd, index='op', columns='device', values='ratio_top5')
    print('\n\nratio_top5\n')
    pprint(df_ratio_top5)
    
    df_ratio_bottom5 = pd.pivot(result_pd, index='op', columns='device', values='ratio_bottom5')
    print('\n\nratio_bottom5\n')
    pprint(df_ratio_bottom5)
    
    
    df_prob = pd.pivot(result_pd, index='op', columns='device', values='prob')
    print('\n\ndiff_prob\n')
    pprint(df_prob)
    

    # result_pd.to_csv('./result_gpu_base/m1_gpu/posit10/result.csv')
    # df_ratio_mean.to_csv('./result_gpu_base/m1_gpu/posit10/result_ratio_mean.csv')
    # df_ratio_top5.to_csv('./result_gpu_base/m1_gpu/posit10/result_ratio_top5.csv')
    # df_ratio_bottom5.to_csv('./result_gpu_base/m1_gpu/posit10/result_ratio_bottom5.csv')
    # df_prob.to_csv('./result_gpu_base/m1_gpu/posit10/result_prob.csv')
    
    
    