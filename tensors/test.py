import pickle
import torch
import torch.nn as nn
import numpy as np

ranges = ['1', '10', '10(-1)', '10(-2)', '10(-3)', 'negat10', 'posit10']
ops = ['softmax', 'GeLU', 'tanh', 'swish']
for range in ranges:
    print("*"*40)
    print(f'\n\n{range}')  
    with open(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/tensors/input_tensor/input_tensor_{range}.pickle', 'rb') as fr:
        input = pickle.load(fr)
        print(f'input : {input[0][0][:10]}\n\n')
    for op in ops:
        with open(f'/home/rudfhr0314/HYU/BDSL/operation_profiling/tensors/tpu_output/{range}/tpu_torch_{op}_output.pickle', 'rb') as fr:
            output = pickle.load(fr)
            print(f'{op}_output : {output[0][0][:5]}\n\n')
      
      
      
#---------------------------------------------------

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# data = torch.tensor(np.array([3.2, 6.1, 0.02, 0.156]))
# data = data.to(device)

# print("-"*40)
# print('\n\n')
# model = nn.Softmax(dim=0)
# model.to(device)
# softmax_output = model(data)
# print(f"softmax_output : {softmax_output}")