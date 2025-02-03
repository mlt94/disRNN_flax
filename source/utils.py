from IPython import embed
from flax import nnx
import jax
from jax import random

'''From the Flax documentation (see "Evolution from Flax Linen to NNX), though removed the dropout'''
class Block(nnx.Module):
  def __init__(self, input_dim, features, rngs):
    self.linear = nnx.Linear(input_dim, features, rngs=rngs)
    
  def __call__(self, x: jax.Array):  
    x = self.linear(x)
    x = jax.nn.relu(x)
    return x   
  
class MLP(nnx.Module):
  def __init__(self, features, num_layers, rngs):
    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def create_block(rngs: nnx.Rngs):
      return Block(features, features, rngs=rngs)

    self.blocks = create_block(rngs)
    self.num_layers = num_layers

  def __call__(self, x):
    @nnx.split_rngs(splits=self.num_layers)
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def forward(x, model):
      x = model(x)
      return x

    return forward(x, self.blocks)


#key = random.PRNGKey(5)
#x = random.uniform(key, shape=(500,10))
#model = MLP(10, num_layers=5, rngs=nnx.Rngs(0))
