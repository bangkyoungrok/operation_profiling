import torch
import torch.nn as nn
import os
import pickle
import numpy as np

print("output start!!!")
# ranges = ['1', '10', '10(-1)', '10(-2)', '10(-3)', 'negat10', 'posit10']
ranges = ['-1~-0.3', '-0.3~0.3', '0.3~1']
#ops = ['softmax', 'GeLU', 'tanh', 'swish']
device_name = 'jetson'
for range in ranges:
    print(f"range : {range}")
    with open(f'/home/rudfhr0314/operation_profiling/tensors/input_tensor/input_tensor_{range}.pickle', 'rb') as fr:
        input = pickle.load(fr)
    
    path = f'/home/rudfhr0314/operation_profiling/tensors/{device_name}_output/{range}'
    
    if not os.path.isdir(path):
        os.makedirs(path)
    
    # to tensor    
    device = torch.device('cuda')
    input = torch.Tensor(input)
    input = input.to(device)
            
    # softmax
    model = nn.Softmax(dim=2)
    model.to(device)
    softmax_output = model(input)
    softmax_output = softmax_output.to('cpu')
    
    with open(os.path.join(path,f'{device_name}_torch_softmax_output.pickle'), 'wb') as fw:
        pickle.dump(softmax_output, fw)
        
    #GeLU
    model = nn.GELU()
    model.to(device)
    GeLU_output = model(input)
    GeLU_output = GeLU_output.to('cpu')

    with open(os.path.join(path,f'{device_name}_torch_GeLU_output.pickle'), 'wb') as fw:
        pickle.dump(GeLU_output, fw)
     
        
    #tanh
    tanh_output = torch.tanh(input)
    tanh_output = tanh_output.to('cpu')
    
    with open(os.path.join(path,f'{device_name}_torch_tanh_output.pickle'), 'wb') as fw:
        pickle.dump(tanh_output, fw)
        

    #swish(SiLU)
    model = nn.SiLU()
    model.to(device)
    swish_output = model(input)
    swish_output = swish_output.to('cpu')

    with open(os.path.join(path,f'{device_name}_torch_swish_output.pickle'), 'wb') as fw:
        pickle.dump(swish_output, fw)