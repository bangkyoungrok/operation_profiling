import torch 
import pickle
import pandas as pd
from pprint import pprint

device_list = ['gpu', 'tpu']
op_list = ['exp', 'softmax', 'GeLU', 'tanh', 'sigmoid', 'swish']
result = {}
for op in op_list:
    result[op] = {}
    for device in device_list:
        with open(f"./tensors/{device}_output/{device}_torch_{op}_output.pickle", "rb") as fr:
            result[op][device] = pickle.load(fr)
        
mean_result = []

def re_mean(x):
    mean = torch.mean(x)
    return mean.item()

for op_name, op_val in result.items():
    for device, op_result in op_val.items():
        mean = re_mean(op_result)
        item = {
            'op' : op_name,
            'device' : device,
            'mean' : mean
        }
        mean_result.append(item)
    """
    print(op_name, op_val)
    break
    """
result_pd = pd.DataFrame(mean_result)
pprint(result_pd)


result_pd.to_csv('./tensors/mean.csv')