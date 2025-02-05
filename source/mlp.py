from IPython import embed
from flax import nnx
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from collections.abc import Callable, Iterable

from flax.nnx.nn import initializers

'''
This is largely the part of the implementation that I am least proud of:
FLax does not offer an MLP implementation out-of-the-box, and the only examples in the Examples and Guides section
does not fit for the present case as the dimensions vary from layer to layer.
I therefore choose to make a small adaption to the Haiku MLP, to make it fit for Flax.
The reason I am not proud about it is that I fear the implementation is not native to the way Flax should be used.
The examples use nnx.vmap and nnx.scan as decorators, and I tried to implement this, but as these functions
have very poor documentation in the Flax landscape, I couldn't make it work. Will update this to be more Flax
native when stronger documentation and examples arrive
HAIKU MLP 
1) Input being a tuple of length layers with values being output size for layer i
2) Create a stack of linear layers by looping over input, setting output size as the value i
3) When called, loop over layers and apply x to each
'''

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


