from IPython import embed
from flax import nnx
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from collections.abc import Callable, Iterable

from flax.nnx.nn import initializers

#Flax have not yet implemented an MLP, so I have made a simple one here.

class haiku_adapated_linear(nnx.Module):
  def __init__(self,
              output_size: int,
              ):
    self.input_size = None
    self.output_size = output_size
  def __call__(
      self, inputs: jax.Array
  ):
    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    
    stddev = 1. / np.sqrt(self.input_size)
    w_init = initializers.truncated_normal(stddev=stddev)(jax.random.key(42), (input_size, output_size))
    w = nnx.Param(w_init)

    out = jnp.dot(inputs, w)
    

    b = initializers.zeros_init()(jax.random.key(42), (output_size))
    b = jnp.broadcast_to(b, out.shape)
    out = out + b
    return out
  

class MLP(nnx.Module):
  def __init__(self, 
               output_sizes: Iterable[int], 
               activation: Callable[[jax.Array], jax.Array] = jax.nn.relu
               ):
    

    self.activation = activation
    layers = []
    output_sizes = tuple(output_sizes)
    for index, output_size in enumerate(output_sizes):
      layers.append(haiku_adapated_linear(output_size=output_size))
    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None

  def __call__(
      self,
      inputs: jax.Array      
  ):
    
    num_layers = len(self.layers)
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      out = self.activation(out)
    return out


