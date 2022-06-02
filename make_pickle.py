import numpy as np
import pickle



tensor_10 = np.random.uniform(-10,10,(100,256,256))
tensor_1 = np.random.uniform(-1,1,(100,256,256))
tensor_10_1 = np.random.uniform(-10**(-1),10**(-1),(100,256,256))
tensor_10_2 = np.random.uniform(-10**(-2),10**(-2),(100,256,256))
tensor_10_3 = np.random.uniform(-10**(-3),10**(-3),(100,256,256))
tensor_posit10 = np.random.uniform(0,10,(100,256,256))
tensor_negat10 = np.random.uniform(-10,0,(100,256,256))
tensors_range = ['input_tensor_10', 'input_tensor_posit10', 'input_tensor_negat10', 'input_tensor_1', 'input_tensor_10(-1)','input_tensor_10(-2)', 'input_tensor_10(-3)']

tensors = [tensor_10, tensor_posit10, tensor_negat10, tensor_1, tensor_10_1, tensor_10_2, tensor_10_3]

for ranges, tensor in zip(tensors_range, tensors):
    print(ranges)
    with open(f"/home/rudfhr0314/HYU/BDSL/operation_profiling/tensors/{ranges}.pickle", "wb") as fw:
        pickle.dump(tensor, fw)