import pickle
import torch

import pandas as pd
from pprint import pprint
import numpy as np

import matplotlib.pyplot as plt


def get_diff_prob(x, y, count):
    diff = torch.abs(x-y)
    x_abs = torch.abs(x)
    ratio = torch.div(diff,x_abs)
    # print('x\n',x)
    # print('y\n',y)
    # print('ratio\n',ratio)
    # ratio_min = torch.min(ratio)
    # ratio_max = torch.max(ratio)
    # print(f'ratio_min\n{ratio_min}\n\nratio_max\n{ratio_max}')
    ratio_mean = torch.mean(ratio)
    # print(f'count : {count}')
    # print(f'x.shape : {x.shape}')
    # print(f'y.shape : {y.shape}')
    # print(f'ratio_mean : {ratio_mean}')
    ratio_var = torch.var(ratio)
    ratio_flatten = torch.flatten(ratio)
    ratio_sort = torch.sort(ratio_flatten, descending=True)[0]
    ratio_top5 = ratio_sort[:5]
    ratio_bottom5 = ratio_sort[-5:]
    
    significant_figure = 4 
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
    return ratio_mean.item(), ratio_var.item(), torch.Tensor.tolist(ratio_top5), torch.Tensor.tolist(ratio_bottom5), prob.item()

    
if __name__=='__main__':
    device_list = ['gpu', 'cpu']
    op_list = ['softmax', 'GeLU', 'tanh', 'swish']
    op_result = {}
    ranges = 'unit_1'
    
    for op in op_list:
        op_result[op] = {}
        for device in device_list:
            with open(f"./tensors/{device}_output/{ranges}/{device}_torch_{op}_output.pickle", "rb") as fr:
                op_result[op][device] = pickle.load(fr)
    diff_result = []
    for op_name, op_val in op_result.items():
        for device, op_result_tensor in op_val.items():
            avg = []
            var = []
            top_5 = []
            bottom_5 = []
            prob = []
            for count in range(20):
                # print(f'\n\nop : {op_name}, count : {count}, deive : {device}')
                ratio_mean, ratio_var, ratio_top5, ratio_bottom5 , prob_ = get_diff_prob(op_val['gpu'][count], op_result_tensor[count], count)
                # if count>=43 and count<46:
                #     print(f'\n\nop : {op_name}, count : {count}, deive : {device}')
                #     print('gpu')
                #     print(op_val['gpu'][count][0][:5])
                #     print('m1')
                #     print(op_val['m1'][count][0][:5])
                # print(ratio_mean)
                avg.append(ratio_mean)
                var.append(ratio_var)
                top_5.append(ratio_top5)
                bottom_5.append(ratio_bottom5)
                prob.append(prob_)
            item = {
                'op' : op_name,
                'device' : device,
                'ratio_mean' : avg,
                'ratio_var' : var,
                'ratio_top5' : top_5,
                'ratio_bottom5' : bottom_5,
                'prob' : prob
            }
            
            diff_result.append(item)
                              

    
                    
    result_pd = pd.DataFrame(diff_result)
    pprint(result_pd)


    # ratio_mean
    df_ratio_mean = pd.pivot(result_pd, index='op', columns='device', values='ratio_mean')
    print('\n\nratio_mean\n')
    pprint(df_ratio_mean)
    # pprint(df_ratio_mean['m1'])
    # print(np.shape(df_ratio_mean['m1']['GeLU']))
    
    
    # ratio_var
    df_ratio_var = pd.pivot(result_pd, index='op', columns='device', values='ratio_var')
    print('\n\nratio_var\n')
    pprint(df_ratio_var)
    
    # for idx, row in df_ratio_mean.iterrows(): 
    
    
    # ratio_top5
    df_ratio_top5 = pd.pivot(result_pd, index='op', columns='device', values='ratio_top5')
    print('\n\nratio_top5\n')
    pprint(df_ratio_top5)
    
    
    # ratio_bottom5
    df_ratio_bottom5 = pd.pivot(result_pd, index='op', columns='device', values='ratio_bottom5')
    print('\n\nratio_bottom5\n')
    pprint(df_ratio_bottom5)
    
    
    # prob
    df_prob = pd.pivot(result_pd, index='op', columns='device', values='prob')
    # print('\n\ndiff_prob\n')
    # pprint(df_prob)
    # print(df_prob['m1']['GeLU'])
    # -------------------

    result_pd.to_csv(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/result_gpu_base/cpu/{ranges}/result.csv')
    df_ratio_mean.to_csv(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/result_gpu_base/cpu/{ranges}/df_ration_mean.csv')
    df_ratio_var.to_csv(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/result_gpu_base/cpu/{ranges}/df_ratio_var.csv')
    df_ratio_top5.to_csv(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/result_gpu_base/cpu/{ranges}/df_ratio_top5.csv')
    df_ratio_bottom5.to_csv(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/result_gpu_base/cpu/{ranges}/df_ratio_bottom5.csv')
    df_prob.to_csv(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/result_gpu_base/cpu/{ranges}/df_prob.csv')   
    
    
    # unit_1에서,
        # m1 GeLU에서 -10~-6까지 숫자가 너무 작아서 ratio=0이 나온다. ex) gpu_output=-2.8087e-15, m1_output=-0. --> |diff|/gpu_output=1.0
        # -6~-5는 m1_output=0이 섞여 있다.
    for idx, op in enumerate(op_list):
        # if op=='GeLU':
        #     plt.subplot(2,1,1)
        #     plt.title(op)
        #     x = np.linspace(-10, 10, 20, endpoint=False)+0.5
        #     plt.bar(x, df_ratio_mean['2080Ti'][op], alpha=0.5)
        #     plt.subplot(2,1,2)
        #     plt.bar(x, np.pad(df_ratio_mean['2080Ti']['GeLU'][6:], (6,0), 'constant', constant_values=0), alpha=0.5)   
        #     # print('\n\n',np.pad(df_ratio_mean['m1']['GeLU'][5:], (5,0), 'constant', constant_values=0))
        #     # print('\n\n',np.shape(np.pad(df_ratio_mean['m1']['GeLU'][5:], (5,0), 'constant', constant_values=0)))
        #     plt.show()
        #     plt.savefig(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/graph/2080Ti/{ranges}/{op}_mean.png')
        #     continue
        plt.figure(idx)
        plt.title(f'{op}')
        x = np.linspace(-10, 10, 20, endpoint=False)+0.5
        plt.bar(x, df_ratio_mean['cpu'][op], alpha=0.5)
        plt.show()
        plt.savefig(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/graph/cpu/{ranges}/{op}_mean.png')
    
    # print('\n\ndf_ratio_mean_GeLU\n')
    # print(df_ratio_mean['tpu']['GeLU']) 
    
    

        
    # for idx, op in enumerate(op_list):
    #     # if op=='GeLU':
    #     #     plt.subplot(2,1,1)
    #     #     plt.title(op)
    #     #     x = np.linspace(-10, 10, 2000, endpoint=False)+0.005
    #     #     plt.plot(x, df_ratio_mean['m1'][op], alpha=0.5)
    #     #     plt.subplot(2,1,2)
    #     #     plt.plot(x, np.pad(df_ratio_mean['m1']['GeLU'][600:], (600,0), 'constant', constant_values=0), alpha=0.5)   
    #     #     # print('\n\n',np.pad(df_ratio_mean['m1']['GeLU'][5:], (5,0), 'constant', constant_values=0))
    #     #     # print('\n\n',np.shape(np.pad(df_ratio_mean['m1']['GeLU'][5:], (5,0), 'constant', constant_values=0)))
    #     #     plt.show()
    #     #     plt.savefig(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/graph/unit_10(-2)/{op}_mean.png')
    #     #     continue
    #     plt.figure(idx)
    #     plt.title(f'{op}')
    #     x = np.linspace(-10, 10, 200, endpoint=False)+0.05
    #     plt.plot(x, df_ratio_mean['tpu'][op], alpha=0.5)
    #     plt.show()
    #     plt.savefig(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/graph/tpu/{ranges}/{op}_mean.png')

        