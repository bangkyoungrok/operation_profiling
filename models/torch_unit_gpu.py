import torch
import torch.nn as nn
import numpy as np
import pickle

with open("./tensors/input_tensor/input_tensor_unit_1.pickle", "rb") as fr:
    input = pickle.load(fr)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_input = torch.Tensor(input)
torch_input = torch_input.to(device)

