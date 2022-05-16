import pickle
import tensorflow as tf

with open("../tensors/input_tensor.pickle", "rb") as fr:
    input = pickle.load(fr)
    
#-------------------------------------------------------------------------------------------------

# tensorflow

# softmax

output = tf.nn.softmax(input)
print("-"*20)
print(f"tf_softmax_output : {output}") # tf_softmax_output : [[0.45850172 0.54149828]]

# GeLU

output = tf.nn.gelu(input)
print(f"tf_GeLU_output : {output}")

# tanh

output = tf.nn.tanh(input)
print(f"tf_tanh_output : {output}")

# sigmoid

output = tf.nn.sigmoid(input)
print(f"tf_sigmoid_output : {output}")

# swish(SiLU)

output = tf.nn.silu(input)
print(f"tf_swish_output : {output}")
