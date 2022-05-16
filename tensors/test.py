import pickle

with open("/home/rudfhr0314/HYU/BDSL/operation_profiling/tensors/gpu_output/gpu_torch_softmax_output.pickle", "rb") as fr:
    input = pickle.load(fr)
    


with open("/home/rudfhr0314/HYU/BDSL/operation_profiling/tensors/input_tensor.pickle", "rb") as fr:
    input = pickle.load(fr)
    
print(input)