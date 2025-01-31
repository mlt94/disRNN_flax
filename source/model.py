from flax import nnx
import jax
import numpy as np
from collections.abc import Callable
from flax.typing import (
    Dtype,
    Initializer,
    Shape
)
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import Linear
from flax.nnx.nn.activations import sigmoid
from flax.nnx.nn.activations import tanh
from typing import Any, TypeVar
from flax.nnx import filterlib, rnglib

A = TypeVar("A")
Array = jax.Array
Output = Any
Carry = Any
default_kernel_init = initializers.lecun_normal()
#from flax import nnx
#from flax.nnx.nn.recurrent import GRUCell
from flax.nnx.nn.recurrent import RNN
#from flax.nnx.nn.recurrent import RNNCellBase
import jax.numpy as jnp
import jax

from IPython import embed


class GRUCell(RNN):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        gate_fn: Callable[..., Any] = sigmoid,
        activation_fn: Callable[..., Any] = tanh,
        kernel_init: Initializer = default_kernel_init,
        recurrent_kernel_init: Initializer = initializers.orthogonal(),
        bias_init: Initializer = initializers.zeros_init(),
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        carry_init: Initializer = initializers.zeros_init(),
        rngs: rnglib.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.rngs = rngs

        # Combine input transformations into a single linear layer
        self.dense_i = Linear(
        in_features=in_features,
        out_features=3 * hidden_features,  # r, z, n
        use_bias=True,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        rngs=rngs,
        )

        self.dense_h = Linear(
        in_features=hidden_features,
        out_features=3 * hidden_features,  # r, z, n
        use_bias=False,
        kernel_init=self.recurrent_kernel_init,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        rngs=rngs,
        )

    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:  # type: ignore[override]
        h = carry
        
        # Compute combined transformations for inputs and hidden state
        x_transformed = self.dense_i(inputs)
        h_transformed = self.dense_h(h)
        embed()
        
        # Split the combined transformations into individual components
        xi_r, xi_z, xi_n = jnp.split(x_transformed, 3, axis=-1)
        hh_r, hh_z, hh_n = jnp.split(h_transformed, 3, axis=-1)

        # Compute gates
        r = self.gate_fn(xi_r + hh_r)
        z = self.gate_fn(xi_z + hh_z)

        # Compute n with an additional linear transformation on h
        n = self.activation_fn(xi_n + r * hh_n)

        # Update hidden state
        new_h = (1.0 - z) * n + z * h
        return new_h, new_h


    def initialize_carry(self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None) -> Array:  # type: ignore[override]
        batch_dims = input_shape[:-1]
        if rngs is None:
            rngs = self.rngs
        mem_shape = batch_dims + (self.hidden_features,)
        h = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
        return h
    @property
    def num_feature_axes(self) -> int:
        return 1


class own_rnn(nnx.RNN):
    def __init__(self, din:int, dmid:int, rngs=nnx.Rngs):
        self.cell = GRUCell(in_features=din, hidden_features=dmid, rngs=rngs)
        self.linear = nnx.Linear(in_features = dmid, out_features=1,  rngs=rngs)
    def __call__(self, x):
        carry = self.cell.initialize_carry((x.shape[:-1]))
        scan_fn = lambda carry, cell, x: cell(carry, x)
        carry, y = nnx.scan(
            scan_fn, in_axes=(nnx.Carry, None, 1), out_axes=(nnx.Carry, 1)
        )(carry, self.cell, x) 
        x = self.linear(y)
        return jax.nn.sigmoid(x)




























# class own_rnn(nnx.Module):
#     def __init__(self, input_size:int, hidden_size:int, rngs: nnx.Rngs):
#         self.cell = GRUCell(in_features=input_size, hidden_features=hidden_size, rngs=rngs)
#         self.rnn = RNN(self.cell, time_major=True) #the dataset has dimensions time_steps, sequences, features, so time is the major axis
#         self.classify = nnx.Linear(hidden_size, 1, rngs=rngs) #output classes

#     def __call__(self, x): 
#         x = self.rnn(x)
#         return jax.nn.sigmoid(self.classify(x))
    





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
    
