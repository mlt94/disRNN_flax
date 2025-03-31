from flax import nnx
import jax
import jax.numpy as jnp
from flax.nnx.nn import initializers
import optax
import numpy as np

''' 
Hej Nikolas,
Jeg har fundet noget i Flax, som jeg er i tvivl om er en bug i deres kode eller om jeg bruger biblioteket forkert. Kunne virkelig godt bruge et par øjne på det hvis du har tid! Håber du nyder Colombia!

Det er en utrolig simpel operation jeg forsøger, som fejler. 
I linje 34 (og 33) forsøger jeg at instantiere en helt standard variabel. 
Class'en kan både initialisers og blive kaldt, men jeg får ikke lov at gøre class'en igennem nnx.Optimizer, linje 54, som fejler pga. 34. 
Hvis jeg kalder 34 som nnx.Param kan koden godt køre (nnx.Param(nnx.sigmoid(0.5)), det er kun hvis variablen ikke er et parameter, at det fejler.

Giver det mening for dig, at man virkelig ikke kan instantiere helt standard variabler i en klasses init? Eller tror du det er en fejl i source koden fra Flax's side?

Alt vel,
Tak min ven
Martin
'''


class test_linear(nnx.Module):
  def __init__(self,
              output_size: int,
              rngs = nnx.Rngs,
              ):
    self.input_size = None
    self.output_size = output_size
    self.rngs = rngs
    #self.value_init = initializers.truncated_normal(1)(rngs.params(), (1, output_size)) #this is the culprit!
    self.value_init = nnx.sigmoid(0.5) #this is the culprit!
  def __call__(
      self, inputs: jax.Array
  ):
    # input_size = self.input_size = inputs.shape[-1]
    # output_size = self.output_size
    # dtype = inputs.dtype
    # key = self.rngs.params()

    # stddev = 1. / np.sqrt(self.input_size)
    # w_init = initializers.truncated_normal(stddev=stddev)(key, (input_size, output_size))
    # w = nnx.Param(w_init, dtype=jnp.float32)
    # out = jnp.dot(inputs, w.value)
    # b = initializers.zeros_init()(key, (output_size))
    # b = jnp.broadcast_to(b, out.shape)
    # out = out + b
    return 
  
x = jnp.ones((300,5))
model = test_linear(1, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3)) #this is the culprit!
''' Linje 54 giver fejlen:
ValueError: Arrays leaves are not supported, at 'value_init': 0.622459352016449'''

@nnx.jit
def train(x):
  y = model(x)
  return y

y = train(x)

''' Hele fejlen:

File "/home/mlut/disRNN_flax/for_nicholas.py", line 40, in <module>
    optimizer = nnx.Optimizer(model, optax.adam(1e-3)) #this is the culprit!
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mlut/disRNN_flax/disrnn_venv/lib/python3.11/site-packages/flax/nnx/object.py", line 79, in __call__
    return _graph_node_meta_call(cls, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mlut/disRNN_flax/disrnn_venv/lib/python3.11/site-packages/flax/nnx/object.py", line 88, in _graph_node_meta_call
    cls._object_meta_construct(node, *args, **kwargs)
  File "/home/mlut/disRNN_flax/disrnn_venv/lib/python3.11/site-packages/flax/nnx/object.py", line 82, in _object_meta_construct
    self.__init__(*args, **kwargs)
  File "/home/mlut/disRNN_flax/disrnn_venv/lib/python3.11/site-packages/flax/nnx/training/optimizer.py", line 193, in __init__
    self.opt_state = _wrap_optimizer_state(tx.init(nnx.state(model, wrt)))
                                                   ^^^^^^^^^^^^^^^^^^^^^
  File "/home/mlut/disRNN_flax/disrnn_venv/lib/python3.11/site-packages/flax/nnx/graph.py", line 1475, in state
    _, state = flatten(node)
               ^^^^^^^^^^^^^
  File "/home/mlut/disRNN_flax/disrnn_venv/lib/python3.11/site-packages/flax/nnx/graph.py", line 404, in flatten
    graphdef = _graph_flatten((), ref_index, flat_state, node)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mlut/disRNN_flax/disrnn_venv/lib/python3.11/site-packages/flax/nnx/graph.py", line 451, in _graph_flatten
    raise ValueError(
ValueError: Arrays leaves are not supported, at 'value_init': 0.622459352016449'''
