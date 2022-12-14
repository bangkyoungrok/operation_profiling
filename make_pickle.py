import numpy as np
import pickle



# tensor_10 = np.random.uniform(-10,10,(100,256,256))
# tensor_1 = np.random.uniform(-1,1,(100,256,256))
# tensor_10_1 = np.random.uniform(-10**(-1),10**(-1),(100,256,256))
# tensor_10_2 = np.random.uniform(-10**(-2),10**(-2),(100,256,256))
# tensor_10_3 = np.random.uniform(-10**(-3),10**(-3),(100,256,256))
# tensor_posit10 = np.random.uniform(0,10,(100,256,256))
# tensor_negat10 = np.random.uniform(-10,0,(100,256,256))

# ------------------ unit ------------------------
# tensor_unit_1 = np.random.uniform(-10, -9, (1, 256, 256)) # -10~10에서 단위 1로 random tensor 생성
# for unit in range(19):
#     item = np.random.uniform(-9+unit, -8+unit, (1, 256, 256))
#     tensor_unit_1 = np.append(tensor_unit_1, item, axis=0)

# tensor_unit_10_1 = np.random.uniform(-10, -9.9, (1, 256, 256))
# for unit in range(199):
#     item = np.random.uniform(-9.9+0.1 * unit, -9.8+0.1 * unit, (1, 256, 256))
#     tensor_unit_10_1 = np.append(tensor_unit_10_1, item, axis=0)
    
tensor_unit_10_2 = np.random.uniform(-10, -9.99, (1, 256, 256))
for unit in range(1999):
    item = np.random.uniform(-9.99+0.01 * unit, -9.98+0.01 * unit, (1, 256, 256))
    tensor_unit_10_2 = np.append(tensor_unit_10_2, item, axis=0)

# ------------------------------------------------ 
tensors_range = ['tensor_unit_10_2']

tensors = [tensor_unit_10_2]

for ranges, tensor in zip(tensors_range, tensors):
    print(ranges)
    with open(f"/home/rudfhr0314/HYU/BDSL/operation_profiling/tensors/input_tensor/input_tensor_unit_10_2.pickle", "wb") as fw:
        pickle.dump(tensor, fw)