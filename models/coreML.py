import coremltools as ct

import torch

import pickle
import numpy as np
"""
model = torch.nn.Softmax()
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('torch_softmax.pt') # Save


mlmodel = ct.convert("torch_softmax.pt", inputs=[ct.TensorType(shape=(100,256,256))], compute_units=ct.ComputeUnit.ALL)

#print(mlmodel)

"""


model = torch.nn.Softmax(dim = 2)
model.eval()
with open("../tensors/input_tensor.pickle", "rb") as fr:
    input = pickle.load(fr)
input = torch.rand(1,2,2)
traced_model = torch.jit.trace(model, input)
traced_model.save("torch_softmax.pt", inputs=[ct.TensorType(shape=(1,2,2))])


#######################

mlmodel = ct.convert("torch_softmax.pt", inputs=[ct.TensorType(shape=(1,2,2))], compute_units=ct.ComputeUnit.ALL)

predictions = mlmodel.predict(input)

print(predictions)
print("*"*20 )

