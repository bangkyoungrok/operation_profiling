import pickle
import torch.nn as nn
import torch

with open("data.pickle", "rb") as fr:
    input = pickle.load(fr)

#-------------------------------------------------------------------------------------------------

# pytorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_input = torch.Tensor(input)
torch_input = torch_input.to(device)
#print(f"tensor_input : {torch_input}") 

# softmax

m = nn.Softmax(dim = 2)
torch_softmax_output = m(torch_input)
torch_softmax_output = torch_softmax_output.to('cpu')
print(f"torch_softmax_output : {torch_softmax_output}") 

with open("output/gpu_torch_softmax_output.pickle", "wb") as fw:
    pickle.dump(torch_softmax_output, fw)

# GeLU

m = nn.GELU()
torch_GeLU_output = m(torch_input)
torch_GeLU_output = torch_GeLU_output.to('cpu')
print(f"torch_GeLU_output : {torch_GeLU_output}") 

with open("output/gpu_torch_GeLU_output.pickle", "wb") as fw:
    pickle.dump(torch_GeLU_output, fw)
    
# tanh

torch_tanh_output = torch.tanh(torch_input)
torch_tanh_output = torch_tanh_output.to('cpu')
print(f"torch_tanh_output : {torch_tanh_output}") 

with open("output/gpu_torch_tanh_output.pickle", "wb") as fw:
    pickle.dump(torch_tanh_output, fw)
    
# sigmoid

torch_sigmoid_output = torch.sigmoid(torch_input)
torch_sigmoid_output = torch_sigmoid_output.to('cpu')
print(f"torch_sigmoid_output : {torch_sigmoid_output}")

with open("output/gpu_torch_sigmoid_output.pickle", "wb") as fw:
    pickle.dump(torch_sigmoid_output, fw)
    
# swish(SiLU)

m = nn.SiLU()
torch_swish_output = m(torch_input)
torch_swish_output = torch_swish_output.to('cpu')
print(f"torch_swish_output : {torch_swish_output}") 

with open("output/gpu_torch_swish_output.pickle", "wb") as fw:
    pickle.dump(torch_swish_output, fw)
    