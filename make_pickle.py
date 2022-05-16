import pickle
import numpy as np

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

np.random.seed(0)
input = np.random.rand(100, 256, 256)
# print(f"seed(0)_input : {input}") # seed(0)_input : [[0.5488135  0.71518937]]

with open("data.pickle", "wb") as fw:
    pickle.dump(input, fw)
