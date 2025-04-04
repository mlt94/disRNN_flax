from IPython import embed
from flax import nnx
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from collections.abc import Callable, Iterable

from flax.nnx.nn import initializers

#Flax have not yet implemented an MLP, so I have made a simple one here.
  
class MLP_loop(nnx.Module):
  def __init__(self, 
               rngs = nnx.Rngs,
               ):
    
    self.rngs = rngs
    self.mlp_linear1_loop = nnx.Linear(in_features=7, out_features=5, rngs=rngs)
    self.mlp_linear2_loop = nnx.Linear(in_features=5, out_features=5, rngs=rngs)
    self.mlp_linear3_loop = nnx.Linear(in_features=5, out_features=5, rngs=rngs)

  def __call__(
      self,
      inputs: jax.Array      
  ):
    out = self.mlp_linear1_loop(inputs)
    out = jax.nn.relu(out)
    out = self.mlp_linear2_loop(out)
    out = jax.nn.relu(out)
    out = self.mlp_linear3_loop(out)
    return out
  

class MLP_choice(nnx.Module):
  def __init__(self, 
              rngs = nnx.Rngs,
              ):
    
    self.rngs = rngs
    self.mlp_linear1_choice = nnx.Linear(in_features=5, out_features=5, rngs=rngs)
    self.mlp_linear2_choice = nnx.Linear(in_features=5, out_features=5, rngs=rngs)
    self.mlp_linear3_choice = nnx.Linear(in_features=5, out_features=5, rngs=rngs)

  def __call__(
      self,
      inputs: jax.Array      
  ):
    out = self.mlp_linear1_choice(inputs)
    out = jax.nn.relu(out)
    out = self.mlp_linear2_choice(out)
    out = jax.nn.relu(out)
    out = self.mlp_linear3_choice(out)
    return out