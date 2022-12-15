import pickle
import torch.nn as nn
import torch
import numpy as np
import os

ranges = ['-1~-0.3', '-0.3~0.3', '0.3~1']

for range in ranges:
    with open(f"/workspace/tensors/input_tensor/input_tensor_{range}.pickle", "rb") as fr:
        input = pickle.load(fr)
    print(input.shape)

    #-------------------------------------------------------------------------------------------------
    path = f"/workspace/tensors/gpu_output/{range}"
    if not os.path.isdir(path):
        os.makedirs(path)

    # pytorch

    device = torch.device("cuda")

    torch_input = torch.Tensor(input)
    torch_input = torch_input.to(device)
    print(device)
    

    # softmax

    m = nn.Softmax(dim = 2)
    torch_softmax_output = m(torch_input)
    torch_softmax_output = torch_softmax_output.to('cpu')
    print(torch_softmax_output.shape)
    #print(f"torch_softmax_output : {torch_softmax_output}") 

    with open(f"/workspace/tensors/gpu_output/{range}/gpu_torch_softmax_output.pickle", "wb") as fw:
        pickle.dump(torch_softmax_output, fw)
        
    # GeLU

    m = nn.GELU()
    torch_GeLU_output = m(torch_input)
    torch_GeLU_output = torch_GeLU_output.to('cpu')
    print(torch_GeLU_output.shape)
    print(f"torch_GeLU_output : {torch_GeLU_output[0][0][0]}") 

    with open(f"/workspace/tensors/gpu_output/{range}/gpu_torch_GeLU_output.pickle", "wb") as fw:
        pickle.dump(torch_GeLU_output, fw)
        
    # tanh

    torch_tanh_output = torch.tanh(torch_input)
    torch_tanh_output = torch_tanh_output.to('cpu')
    print(torch_tanh_output.shape)
    #print(f"torch_tanh_output : {torch_tanh_output}") 

    with open(f"/workspace/tensors/gpu_output/{range}/gpu_torch_tanh_output.pickle", "wb") as fw:
        pickle.dump(torch_tanh_output, fw)
        


    # swish(SiLU)

    m = nn.SiLU()
    torch_swish_output = m(torch_input)
    torch_swish_output = torch_swish_output.to('cpu')
    print(torch_swish_output.shape)
    #print(f"torch_swish_output : {torch_swish_output}") 

    with open(f"/workspace/tensors/gpu_output/{range}/gpu_torch_swish_output.pickle", "wb") as fw:
        pickle.dump(torch_swish_output, fw)
        