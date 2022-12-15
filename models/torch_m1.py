import torch
import torch.nn as nn

import pickle
import numpy as np


ranges = ['1', '10', '10(-1)', '10(-2)', '10(-3)', 'negat10', 'posit10']
#ops = ['softmax', 'GeLU', 'tanh', 'swish']

for range in ranges:
                
    with open(f'/Users/bang-kyoungrok/Desktop/HYU/BDSL/operation_profiling/tensors/input_tensor/input_tensor_{range}.pickle', 'rb') as fr:
        input = pickle.load(fr)
    
    # to tensor    
    mps_device = torch.device('mps')
    input = torch.Tensor(input)
    input = input.to(mps_device)
            
            
    # softmax
    model = nn.Softmax(dim=2)
    model.to(mps_device)
    softmax_output = model(input)
    softmax_output = softmax_output.to('cpu')

    with open(f'/Users/bang-kyoungrok/Desktop/HYU/BDSL/operation_profiling/tensors/output_tensor/{range}/m1_torch_softmax_output.pickle', 'wb') as fw:
        pickle.dump(softmax_output, fw)
        

    #GeLU
    model = nn.GELU()
    model.to(mps_device)
    GeLU_output = model(input)
    GeLU_output = GeLU_output.to('cpu')

    with open(f'/Users/bang-kyoungrok/Desktop/HYU/BDSL/operation_profiling/tensors/output_tensor/{range}/m1_torch_GeLU_output.pickle', 'wb') as fw:
        pickle.dump(GeLU_output, fw)
     
        
    #tanh
    tanh_output = torch.tanh(input)
    tanh_output = tanh_output.to('cpu')
    with open(f'/Users/bang-kyoungrok/Desktop/HYU/BDSL/operation_profiling/tensors/output_tensor/{range}/m1_torch_tanh_output.pickle', 'wb') as fw:
        pickle.dump(tanh_output, fw)
        

    #swish(SiLU)
    model = nn.SiLU()
    model.to(mps_device)
    swish_output = model(input)
    swish_output = swish_output.to('cpu')

    with open(f'/Users/bang-kyoungrok/Desktop/HYU/BDSL/operation_profiling/tensors/output_tensor/{range}/m1_torch_swish_output.pickle', 'wb') as fw:
        pickle.dump(swish_output, fw)