from flax import nnx
from flax.nnx.nn.recurrent import GRUCell
from flax.nnx.nn.recurrent import RNN

import jax.numpy as jnp
import jax


from IPython import embed


class own_rnn(nnx.Module):
    def __init__(self, input_size:int, hidden_size:int, rngs: nnx.Rngs):
        self.cell = GRUCell(in_features=input_size, hidden_features=hidden_size, rngs=rngs)
        self.rnn = RNN(self.cell, time_major=True) #the dataset has dimensions time_steps, sequences, features, so time is the major axis
        self.classify = nnx.Linear(hidden_size, 1, rngs=rngs) #output classes

    def __call__(self, x): 
        x = self.rnn(x)
        return jax.nn.sigmoid(self.classify(x))
    












































# class RNNCell(nnx.Module):
#     def __init__(self, input_size, hidden_size, rngs):
#         self.linear = nnx.Linear(hidden_size + input_size, hidden_size, rngs=rngs)
#         self.hidden_size = hidden_size

#     def __call__(self, carry, x):
#         x = jnp.concatenate([carry, x], axis=-1)
#         x = self.linear(x)
#         x = jax.nn.relu(x)
#         return x, x
    
#     def initial_state(self, n_steps: int):
#         return jnp.zeros((n_steps, self.hidden_size))
    

# class RNN(nnx.Module):
#     def __init__(self, input_size:int, hidden_size:int, rngs: nnx.Rngs):
#         self.hidden_size = hidden_size
#         self.cell = RNNCell(input_size, self.hidden_size, rngs=rngs)
#         self.classify = nnx.Linear()

#     def __call__(self, x):
#         scan_fn = lambda carry, cell, x: cell(carry, x)
#         carry = self.cell.initial_state(x.shape[0]) #n_steps x features

#         carry, y = nnx.scan(
#             scan_fn, in_axes=(nnx.Carry, None, 1), out_axes=(nnx.Carry, 1)#in_axes and out_axes are positional arguments for nnx.scan
#         )(carry, self.cell, x) #nnx.scan is a transformation that takes as input a function
#         return y
    
